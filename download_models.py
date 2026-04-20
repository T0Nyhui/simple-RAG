import os
# 确保镜像站生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id="jinaai/jina-embeddings-v4",
    local_dir="./models/jina-v4",
    local_dir_use_symlinks=False,
    resume_download=True,
    token=None # 如果模型是公开的则不需要 token
)
print("下载完成！")