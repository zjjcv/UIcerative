import re

def parse_medical_report(file_path):
    # 读取文件并处理BOM
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    report = {}
    current_key = None
    indent_pattern = re.compile(r'^\s{4}')  # 匹配4个空格的缩进

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 处理多行值的情况
        if indent_pattern.match(line):
            if current_key and current_key in report:
                report[current_key] += ' ' + line.strip()
            continue

        # 分割键值对
        if '：' in line:
            key, value = line.split('：', 1)
            key = key.strip()
            value = value.strip()
            
            if value:  # 只保留有值的字段
                report[key] = value
                current_key = key
        else:
            current_key = None

    # 转换为自然语言描述
    text_parts = []
    for key, value in report.items():
        if key in ['镜下所见', '镜下诊断']:  # 特殊处理多行字段
            text_parts.append(f"{key}：{value}")
        else:
            text_parts.append(f"{key}为{value}")

    return '。'.join(text_parts) + '。'

# 使用示例
if __name__ == "__main__":
    text_embedding_input = parse_medical_report(r'data\CD\1\1.txt')
    print("整理后的文本：\n", text_embedding_input)
    print("\n可直接用于文本嵌入模型")