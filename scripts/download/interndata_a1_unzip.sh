src="/apdcephfs_cq12/share_1150325/hetolin/datasets/InternData-A1-zip"
dst="/apdcephfs_cq12/share_1150325/hetolin/datasets/InternData-A1-full"


rsync -a -f"+ */" -f"- *" "$src/" "$dst/"

export src dst
find "$src" -type f \( -name "*.tar.gz" -o -name "*.tgz" \) | \
xargs -P 16 -I{} bash -c '
    relpath="${1#"$src"/}"
    outdir="$dst/$(dirname "$relpath")"
    pkgbase=$(basename "$1" .tar.gz)
    pkgbase=${pkgbase%.tgz}
    subdir="$outdir/$pkgbase"
    mkdir -p "$subdir"
    echo "解压: $1 到 $subdir"
    tar -xzf "$1" -C "$subdir"
' _ {}

## 1. 复制目录结构
#rsync -a -f"+ */" -f"- *" "$src/" "$dst/"
#
## 2. 复制所有压缩包到新的同目录
#cd "$src"
#find . -type f \( -name "*.tar.gz" -o -name "*.tgz" \) | while read f; do
#    mkdir -p "$dst/$(dirname "$f")"
#    cp "$src/$f" "$dst/$f"
#done
#
## 3. 递归解压所有包
#cd "$dst"
#while true; do
#    found=$(find . -type f \( -name "*.tar.gz" -o -name "*.tgz" \))
#    [ -z "$found" ] && break
#    echo "$found" | while read f; do
#        echo "解压: $f"
#        tar -xzf "$f" -C "$(dirname "$f")"
#        # rm "$f"  # 如要删除原包，取消注释
#    done
#done



#cd "$dst"
#while true; do
#    found=$(find . -type f \( -name "*.tar.gz" -o -name "*.tgz" \))
#    [ -z "$found" ] && break
#    # 多进程并行解压
#    echo "$found" | parallel -j 64 'echo "解压: {}"; tar -xzf {} -C {= s:/[^/]+$:: =}'
#    # 解压后要删除包可以这样（放开 rm）：
#    # echo "$found" | parallel -j 8 'tar -xzf {} -C {= s:/[^/]+$:: =} && rm {}'
#done