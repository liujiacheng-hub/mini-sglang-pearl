from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary_actor():
            ack_queue.put(f"{args.tp_info.actor} Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if scheduler.tp_info.is_primary_actor():
                print()  # for a clean newline after ^C
                logger.info(f"{args.tp_info.actor} Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        enable_pearl = server_args.enable_pearl
        if enable_pearl:
            target_tp_size = server_args.tp_size_target
            draft_tp_size = server_args.tp_size_draft
            assert world_size == target_tp_size + draft_tp_size
            assert server_args.draft_model_path is not None
        else:
            target_tp_size = world_size
            draft_tp_size = 0

        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            actor = "target" if i < target_tp_size else "draft"
            local_rank = i if i < target_tp_size else i - target_tp_size
            local_size = target_tp_size if i < target_tp_size else draft_tp_size
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size, actor, local_rank, local_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"{actor}-minisgl-TP{local_rank}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        
        
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2

        # TODO(ljc)
        if enable_pearl:
            num_acks = num_tokenizers + 3
        else:
            num_acks = num_tokenizers + 2

        for _ in range(num_acks):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
