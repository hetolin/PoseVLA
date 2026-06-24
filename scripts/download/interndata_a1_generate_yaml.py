import os
import json
from collections import defaultdict


def process_datasets(base_dir, search_sub_dirs, output_dir):
    # --- 1. 准备输出目录 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # --- 2. 扫描数据集并按 [特征名 + 维度] 分组 ---
    groups = defaultdict(list)
    abs_base = os.path.abspath(base_dir)

    print(f"正在扫描目录: {abs_base} ...")
    for sub_dir in search_sub_dirs:
        physical_path = os.path.join(abs_base, sub_dir)
        if not os.path.exists(physical_path):
            print(f"警告: 路径不存在 {physical_path}")
            continue

        for root, dirs, files in os.walk(physical_path):
            if "meta" in dirs and "data" in dirs:
                info_path = os.path.join(root, "meta", "info.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            features_dict = info.get("features", {})

                            # --- 核心修改：构建包含维度的特征签名 ---
                            # 格式: ((feat1, shape1), (feat2, shape2), ...)
                            signature = []
                            for feat_name in sorted(features_dict.keys()):
                                shape = tuple(features_dict[feat_name].get("shape", []))
                                signature.append((feat_name, shape))

                            group_key = tuple(signature)
                            rel_path = os.path.relpath(root, abs_base).replace(os.sep, '/')
                            groups[group_key].append(rel_path)
                    except Exception as e:
                        print(f"读取失败 {info_path}: {e}")
                dirs[:] = []

    if not groups:
        print("未找到任何有效数据集，请检查路径。")
        return

    # 按数据集数量排序
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    # 获取所有出现过的特征名称
    all_feat_names = set()
    for sig in groups.keys():
        for name, shape in sig:
            all_feat_names.add(name)
    union_features = sorted(list(all_feat_names))

    # --- 3. 打印特征差异矩阵 (显示维度) ---
    print("\n" + "=" * 120)
    print(f"{'特征分布矩阵 (显示维度, -: 缺失)':^120}")
    print("=" * 120)

    max_feat_len = max(len(f) for f in union_features) + 2
    first_col_w = max(max_feat_len, 30)

    header = "Feature Name".ljust(first_col_w)
    for i in range(1, len(sorted_groups) + 1):
        header += f"  G{i:02}  "
    print(header)
    print("-" * len(header))

    for f_name in union_features:
        row = f_name.ljust(first_col_w)
        for sig, _ in sorted_groups:
            # 在签名中查找该特征的维度
            feat_shape = next((shape for name, shape in sig if name == f_name), None)
            if feat_shape is not None:
                # 简写维度显示，如 (6,) -> 6, (3, 224, 224) -> 3x224
                shape_str = "x".join(map(str, feat_shape))
                if len(shape_str) > 6: shape_str = shape_str[:5] + ".."
                row += f"{shape_str:^8}"
            else:
                row += f"{'-':^8}"
        print(row)
    print("-" * len(header))

    # --- 4. 生成 YAML 文件 ---
    print("\n" + "=" * 120)
    print(f"{'YAML 生成报告':^120}")
    print("=" * 120)

    for i, (sig, repo_ids) in enumerate(sorted_groups, 1):
        group_id = f"G{i:02}"
        yaml_filename = f"lerobot_group{i:02}.yaml"
        yaml_full_path = os.path.join(output_dir, yaml_filename)

        yaml_content = [
            "defaults:",
            "  - lerobot",
            "",
            f"# Group {group_id}",
            f"# Datasets: {len(repo_ids)}",
            "# Feature Shapes:",
        ]
        # 在注释里写清楚维度，方便调试
        for name, shape in sig:
            yaml_content.append(f"#   - {name}: {list(shape)}")

        yaml_content.append(f'lerobot_dir: "{base_dir}"')
        yaml_content.append("repo_id:")

        for rid in sorted(repo_ids):
            yaml_content.append(f'  - "{rid}"')

        with open(yaml_full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_content))

        print(f"{group_id} -> {yaml_full_path} ({len(repo_ids)} datasets)")

    print(f"\n完成！所有 YAML 已保存在: {output_dir}")


if __name__ == "__main__":
    DATASET_BASE = "mydata"
    SEARCH_PATHS = ["InternData-A1-full/sim_updated_lerobotv30"]
    OUTPUT_CONFIG_DIR = "config/dataset_lerobot"

    process_datasets(DATASET_BASE, SEARCH_PATHS, OUTPUT_CONFIG_DIR)
