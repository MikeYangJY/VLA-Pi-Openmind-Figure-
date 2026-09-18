# 中文学习注释（2026-09-18）；上游 openpi 215abfb217db。
# 此文件仅新增注释；原代码、英文docstring和版权声明保留。
# 定位：文本与状态入口｜π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。
# 阅读顺序：看tokenize的state分支：π0只编码指令，π0.5可把归一化状态分箱后写进文本。其余tokenizer可跳过。
# 重点边界：离散化state不是离散化输出action；π0.5在本仓库仍输出连续动作。
# 本学习目录仅以π0、π0.5及其共用管线为主线；其他模型可跳过。
import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


# 【PaligemmaTokenizer】π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class PaligemmaTokenizer:
    # 【PaligemmaTokenizer.__init__】初始化PaligemmaTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._tokenizer。
    # 输入接口：max_len:int。
    # 内部调用线索：download.maybe_download → path.open → sentencepiece.SentencePieceProcessor → f.read（含分支中的调用，实际路径由条件决定）。
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    # 【PaligemmaTokenizer.tokenize】清理任务文本；有state时将其按归一化区间分成256箱并加入Task/State/Action格式，最后截断或补齐并生成token mask。
    # 输入接口：prompt:str；state:np.ndarray | None。
    # 返回类型：tuple[np.ndarray, np.ndarray]；类型/shape约定需与调用方配套。
    # 内部调用线索：prompt.strip().replace('_', ' ').replace → prompt.strip().replace → prompt.strip → np.digitize → np.linspace（含分支中的调用，实际路径由条件决定）。
    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            # 学习提示：π0.5：把已归一化state按256个区间编码；这是输入状态编码，输出仍是连续动作。
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            # 学习提示：把任务与离散状态放进同一文本前缀，让VLM前缀直接看到state。
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


# 【FASTTokenizer】π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class FASTTokenizer:
    # 【FASTTokenizer.__init__】初始化FASTTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._paligemma_tokenizer、self._fast_tokenizer、self._fast_skip_tokens。
    # 输入接口：max_len:int；fast_tokenizer_path:str。
    # 内部调用线索：download.maybe_download → path.open → sentencepiece.SentencePieceProcessor → f.read → AutoProcessor.from_pretrained（含分支中的调用，实际路径由条件决定）。
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    # 【FASTTokenizer.tokenize】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 prompt.lower().strip().replace → prompt.lower().strip → prompt.lower 追踪具体实现。
    # 输入接口：prompt:str；state:np.ndarray；actions:np.ndarray | None。
    # 返回类型：tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]；类型/shape约定需与调用方配套。
    # 内部调用线索：prompt.lower().strip().replace → prompt.lower().strip → prompt.lower → np.digitize → np.linspace（含分支中的调用，实际路径由条件决定）。
    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        # 学习提示：π0.5：把已归一化state按256个区间编码；这是输入状态编码，输出仍是连续动作。
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    # 【FASTTokenizer.extract_actions】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._paligemma_tokenizer.decode → tokens.tolist → np.zeros 追踪具体实现。
    # 输入接口：tokens:np.ndarray；action_horizon:int；action_dim:int。
    # 返回类型：np.ndarray；类型/shape约定需与调用方配套。
    # 内部调用线索：self._paligemma_tokenizer.decode → tokens.tolist → np.zeros → np.array → self._paligemma_tokenizer.encode（含分支中的调用，实际路径由条件决定）。
    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    # 【FASTTokenizer._act_tokens_to_paligemma_tokens】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array → self._paligemma_tokenizer.vocab_size 追踪具体实现。
    # 输入接口：tokens:np.ndarray | list[int]。
    # 返回类型：np.ndarray；类型/shape约定需与调用方配套。
    # 内部调用线索：isinstance → np.array → self._paligemma_tokenizer.vocab_size（含分支中的调用，实际路径由条件决定）。
    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


# 【BinningTokenizer】π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    # 【BinningTokenizer.__init__】初始化BinningTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._n_bins、self._paligemma_tokenizer、self._fast_skip_tokens。
    # 输入接口：max_len:int；n_bins:int。
    # 内部调用线索：download.maybe_download → path.open → sentencepiece.SentencePieceProcessor → f.read（含分支中的调用，实际路径由条件决定）。
    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    # 【BinningTokenizer.tokenize】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 prompt.lower().strip().replace → prompt.lower().strip → prompt.lower 追踪具体实现。
    # 输入接口：prompt:str；state:np.ndarray；actions:np.ndarray | None。
    # 返回类型：tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]；类型/shape约定需与调用方配套。
    # 内部调用线索：prompt.lower().strip().replace → prompt.lower().strip → prompt.lower → np.digitize → np.linspace（含分支中的调用，实际路径由条件决定）。
    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        # 学习提示：π0.5：把已归一化state按256个区间编码；这是输入状态编码，输出仍是连续动作。
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    # 【BinningTokenizer.extract_actions】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._paligemma_tokenizer.decode → tokens.tolist → np.zeros 追踪具体实现。
    # 输入接口：tokens:np.ndarray；action_horizon:int；action_dim:int。
    # 返回类型：np.ndarray；类型/shape约定需与调用方配套。
    # 内部调用线索：self._paligemma_tokenizer.decode → tokens.tolist → np.zeros → np.array → self._paligemma_tokenizer.encode（含分支中的调用，实际路径由条件决定）。
    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    # 【BinningTokenizer._act_tokens_to_paligemma_tokens】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array → self._paligemma_tokenizer.vocab_size 追踪具体实现。
    # 输入接口：tokens:np.ndarray | list[int]。
    # 返回类型：np.ndarray；类型/shape约定需与调用方配套。
    # 内部调用线索：isinstance → np.array → self._paligemma_tokenizer.vocab_size（含分支中的调用，实际路径由条件决定）。
    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


# 【FSQTokenizer】π0/π0.5主线只需PaligemmaTokenizer：把指令及可选离散状态转成token和padding mask。 本类封装其中的配置、状态或接口；具体入口见下面的方法。
class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    # 【FSQTokenizer.__init__】初始化FSQTokenizer的依赖和内部状态；构造对象本身与后续执行/训练要区分。主要保存：self._max_len、self._params、self._fsq_tokenizer、self._tokenize_fn、self._detokenize_fn。
    # 输入接口：max_len:int；fsq_tokenizer_path:str | None。
    # 内部调用线索：download.maybe_download → os.path.join → os.listdir → tok_path.split → tok_path.rsplit（含分支中的调用，实际路径由条件决定）。
    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    # 【FSQTokenizer.tokenize】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 prompt.lower().strip().replace → prompt.lower().strip → prompt.lower 追踪具体实现。
    # 输入接口：prompt:str；state:np.ndarray；actions:np.ndarray | None。
    # 返回类型：tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]；类型/shape约定需与调用方配套。
    # 内部调用线索：prompt.lower().strip().replace → prompt.lower().strip → prompt.lower → np.digitize → np.linspace（含分支中的调用，实际路径由条件决定）。
    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        # 学习提示：π0.5：把已归一化state按256个区间编码；这是输入状态编码，输出仍是连续动作。
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    # 【FSQTokenizer.extract_actions】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 self._paligemma_tokenizer.decode → tokens.tolist → np.zeros 追踪具体实现。
    # 输入接口：tokens:np.ndarray；action_horizon:int；action_dim:int。
    # 返回类型：np.ndarray；类型/shape约定需与调用方配套。
    # 内部调用线索：self._paligemma_tokenizer.decode → tokens.tolist → np.zeros → np.array → self._paligemma_tokenizer.encode（含分支中的调用，实际路径由条件决定）。
    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    # 【FSQTokenizer._act_tokens_to_paligemma_tokens】本函数位于“文本与状态入口”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 isinstance → np.array → self._paligemma_tokenizer.vocab_size 追踪具体实现。
    # 输入接口：tokens:np.ndarray | list[int]。
    # 返回类型：np.ndarray；类型/shape约定需与调用方配套。
    # 内部调用线索：isinstance → np.array → self._paligemma_tokenizer.vocab_size（含分支中的调用，实际路径由条件决定）。
    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
