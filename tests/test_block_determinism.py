from __future__ import annotations

import os

from replay_exporter.dsp import render_block
from replay_exporter.engine import calculate_gains, parse_and_plan

from .base import ExporterTestCase


class BlockDeterminismTests(ExporterTestCase):
    def test_each_block_independent_of_render_start_position(self):
        plan, sources = parse_and_plan(self.timeline_path)
        gains, previous_gains, _ = calculate_gains(plan, sources)

        # Render every block in reverse order, one at a time, as if resuming at
        # arbitrary block starts. Concatenation must equal the in-order render.
        ordered = bytearray()
        reversed_chunks = {}
        for index in range(plan.block_count()):
            pcm = render_block(plan, index, sources, gains, previous_gains)
            ordered += pcm
        for index in range(plan.block_count() - 1, -1, -1):
            reversed_chunks[index] = render_block(
                plan, index, sources, gains, previous_gains
            )
        rebuilt = bytearray()
        for index in range(plan.block_count()):
            rebuilt += reversed_chunks[index]
        self.assertEqual(bytes(ordered), bytes(rebuilt))

    def test_render_block_is_a_pure_function_of_inputs(self):
        plan, sources = parse_and_plan(self.timeline_path)
        gains, previous_gains, _ = calculate_gains(plan, sources)
        first = render_block(plan, 7, sources, gains, previous_gains)
        second = render_block(plan, 7, sources, gains, previous_gains)
        self.assertEqual(first, second)
        self.assertEqual(len(first), min(plan.block_frames, plan.total_frames - 7 * plan.block_frames) * 4)
