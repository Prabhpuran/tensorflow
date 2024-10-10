/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/transforms/collective_send_recv_combiner.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
namespace xla {

namespace {

// WrapMultipleSendRecvInstructions is a side-effecting function that
// creates a single computation that wraps all the send/recv instructions.
// As a side effect, the function populates the async_start_inputs
// and async_start_input_shapes vectors with the operands and operand shapes of
// the cloned instruction.
HloComputation* WrapMultipleSendRecvInstructions(
    std::vector<HloInstruction*>& send_recv_instructions,
    std::vector<HloInstruction*>& async_start_inputs,
    std::vector<Shape>& async_start_input_shapes,
    HloComputation::Builder& builder, HloModule* module) {
  int operand_counter = 0;
  std::vector<HloInstruction*> new_send_recv_instructions;
  for (auto instruction : send_recv_instructions) {
    std::vector<HloInstruction*> new_operands;
    for (auto operand : instruction->operands()) {
      new_operands.push_back(
          builder.AddInstruction(HloInstruction::CreateParameter(
              operand_counter, operand->shape(),
              absl::StrCat("param", operand_counter))));
      async_start_inputs.push_back(operand);
      async_start_input_shapes.push_back(operand->shape());
      operand_counter++;
    }
    if (instruction->opcode() == HloOpcode::kSend) {
      new_send_recv_instructions.push_back(builder.AddInstruction(
          HloInstruction::CreateSend(new_operands[0], new_operands[1],
                                     instruction->channel_id().value())));
    } else if (instruction->opcode() == HloOpcode::kRecv) {
      new_send_recv_instructions.push_back(
          builder.AddInstruction(HloInstruction::CreateRecv(
              instruction->shape().tuple_shapes(0), new_operands[0],
              instruction->channel_id().value())));
    }
  }
  auto root = builder.AddInstruction(
      HloInstruction::CreateTuple(new_send_recv_instructions));
  return module->AddEmbeddedComputation(builder.Build(root));
}

absl::Status CreateAsyncStartAndAsyncDone(
    std::vector<HloInstruction*>& send_recv_instructions,
    HloComputation* async_computation, HloComputation* computation,
    HloModule* module, std::vector<HloInstruction*>& async_start_inputs,
    std::vector<Shape>& async_start_input_shapes, bool& changed) {
  // Async-start shape consists of (tuple_of_operand_shapes,
  // func_output_shape, s32[]), where s32[] is the context state that is
  // used to keep track of the asynchronous operation. For more details,
  // see https://openxla.org/xla/async_ops.
  Shape async_start_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape(async_start_input_shapes),
       async_computation->root_instruction()->shape(),
       ShapeUtil::MakeScalarShape(S32)});
  auto async_start =
      computation->AddInstruction(HloInstruction::CreateAsyncStart(
          async_start_shape, async_start_inputs, async_computation));
  auto async_done = computation->AddInstruction(HloInstruction::CreateAsyncDone(
      async_computation->root_instruction()->shape(), async_start));
  std::vector<HloInstruction*> replacement_async_dones;
  int async_done_gte_index = 0;
  for (auto instruction : send_recv_instructions) {
    // Create the gte(async-done) instructions to replace send-done/recv-done
    auto unwrapped_async_done =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            instruction->shape(), async_done, async_done_gte_index));
    ++async_done_gte_index;
    // send-done only returns the control-flow token, which is the last element
    // in the unwrapped async-done tuple. recv-done returns the received data
    // and the control-flow token, in that order.
    if (instruction->opcode() == HloOpcode::kSend) {
      replacement_async_dones.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              unwrapped_async_done->shape().tuple_shapes(2),
              unwrapped_async_done, 2)));
    } else if (instruction->opcode() == HloOpcode::kRecv) {
      auto first_element_in_recv_done =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              unwrapped_async_done->shape().tuple_shapes(0),
              unwrapped_async_done, 0));
      auto second_element_in_recv_done =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              unwrapped_async_done->shape().tuple_shapes(2),
              unwrapped_async_done, 2));
      auto recv_done_tuple =
          computation->AddInstruction(HloInstruction::CreateTuple(
              {first_element_in_recv_done, second_element_in_recv_done}));
      replacement_async_dones.push_back(recv_done_tuple);
    }

    for (auto instruction_user : instruction->users()) {
      if (instruction_user->opcode() == HloOpcode::kSendDone ||
          instruction_user->opcode() == HloOpcode::kRecvDone) {
        TF_RETURN_IF_ERROR(instruction_user->ReplaceAllUsesWith(
            replacement_async_dones.back()));
        TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction_user));
        changed = true;
      }
    }
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CollectiveSendRecvCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int wrapped_computation_index = 0;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    // For now we don't transform send/recv instructions in a while loop, since
    // we still need to implement the check for partially pipelined send/recv.
    auto computation_callers = call_graph->GetComputationCallers(computation);
    for (auto caller : computation_callers) {
      if (caller->opcode() == HloOpcode::kWhile) {
        return changed;
      }
    }
    ++wrapped_computation_index;
    std::vector<HloInstruction*> send_recv_instructions;
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kSend &&
          instruction->opcode() != HloOpcode::kRecv) {
        continue;
      }
      send_recv_instructions.push_back(instruction);
    }

    // Create a new computation that wraps the send/recv instructions.
    auto builder = HloComputation::Builder(
        absl::StrCat("wrapped_send_recv_", wrapped_computation_index));
    std::vector<HloInstruction*> async_start_inputs;
    std::vector<Shape> async_start_input_shapes;
    auto async_computation = WrapMultipleSendRecvInstructions(
        send_recv_instructions, async_start_inputs, async_start_input_shapes,
        builder, module);
    TF_RETURN_IF_ERROR(CreateAsyncStartAndAsyncDone(
        send_recv_instructions, async_computation, computation, module,
        async_start_inputs, async_start_input_shapes, changed));
  }
  return changed;
}

}  // namespace xla
