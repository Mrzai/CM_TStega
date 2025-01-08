import json
import os

def fix_coco_json(input_path, output_path):
    """转换COCO数据集的JSON格式，确保兼容性"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 确保images列表存在
    if 'images' not in data:
        raise ValueError("JSON文件缺少'images'字段")
        
    # 转换每个图像条目
    for img in data['images']:
        # 如果存在cocoid但没有id，则复制cocoid到id
        if 'cocoid' in img and 'id' not in img:
            img['id'] = img['cocoid']
            
        # 确保file_path字段存在且正确
        if 'filepath' in img and 'filename' in img:
            # 构建完整的文件路径
            img['file_path'] = os.path.join(img['filepath'], img['filename'])
        elif 'file_name' in img:
            # 如果已经有file_name，确保它包含完整路径
            if not img.get('file_path'):
                img['file_path'] = os.path.join('val2014', img['file_name'])
                
        # 确保所有必要的字段都存在
        if 'id' not in img:
            img['id'] = img.get('cocoid', img.get('image_id', 0))
        if 'file_path' not in img:
            print(f"Warning: image {img.get('id')} missing file_path")
            
    # 保存修改后的JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # 打印一些统计信息
    print(f"处理了 {len(data['images'])} 张图片")
    print("示例图片条目:")
    print(json.dumps(data['images'][0], indent=2))

if __name__ == '__main__':
    input_json = '/data/I2T_data/MSCOCO/dataset_coco.json'
    output_json = '/data/I2T_data/MSCOCO/dataset_coco_fixed.json'
    fix_coco_json(input_json, output_json)
    print("JSON文件已修复") 