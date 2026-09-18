# `src/openpi/shared/download.py` 中文阅读说明

**定位：** 资源加载。

根据本地路径或远程URI查找资源、使用缓存与文件锁，并处理失效缓存。

**建议读法：** maybe_download是入口；按协议选下载方式，再检查权限与时间信息。

**易错点：** 加载策略可能触发大权重下载；源码阅读不需要调用它。

[注释源码](../../../../code/src/openpi/shared/download.py) · [上游固定版本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/shared/download.py)

## 按符号逐段阅读

方法表按源码出现顺序排列；上层调用关系与具体参数见源码中的中文注释。

| 类 / 函数 | 这一段负责什么 |
|---|---|
| [get_cache_dir](../../../../code/src/openpi/shared/download.py#L34) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expandus… → pathlib.Path → os.getenv 追踪具体实现。 |
| [maybe_download](../../../../code/src/openpi/shared/download.py#L45) | 本地路径直接使用；远程路径经过缓存与文件锁后下载，避免多个进程重复写入。 |
| [_download_gsutil](../../../../code/src/openpi/shared/download.py#L125) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 shutil.which → logger.warning → _download_fsspec 追踪具体实现。 |
| [_download_fsspec](../../../../code/src/openpi/shared/download.py#L144) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 fsspec.core.url_to_fs → fs.info → info['name'].endswith 追踪具体实现。 |
| [_set_permission](../../../../code/src/openpi/shared/download.py#L166) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 path.stat → logger.debug → path.chmod 追踪具体实现。 |
| [_set_folder_permission](../../../../code/src/openpi/shared/download.py#L178) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _set_permission 追踪具体实现。 |
| [_ensure_permissions](../../../../code/src/openpi/shared/download.py#L187) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 _setup_folder_permission_between_cache_dir_and_path → os.walk → pathlib.Path 追踪具体实现。 |
| [_ensure_permissions._setup_folder_permission_between_cache_dir_and_path](../../../../code/src/openpi/shared/download.py#L196) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 get_cache_dir → path.relative_to → _set_folder_permission 追踪具体实现。 |
| [_ensure_permissions._set_file_permission](../../../../code/src/openpi/shared/download.py#L208) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 file_path.stat → _set_permission 追踪具体实现。 |
| [_get_mtime](../../../../code/src/openpi/shared/download.py#L232) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 datetime.datetime → time.mktime → date.timetuple 追踪具体实现。 |
| [_should_invalidate_cache](../../../../code/src/openpi/shared/download.py#L252) | 本函数位于“资源加载”路径，负责把调用方输入推进到下一处理阶段。可从其实际调用 local_path.exists → local_path.relative_to → _INVALIDATE_CACHE_DIRS.items 追踪具体实现。 |
