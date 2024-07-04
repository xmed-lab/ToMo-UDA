from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time

def plot_bboxes(image, pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, save_path, ids, gt):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 从Tensor转换为NumPy格式，并进行适当的变换
    image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    img_ori = Image.fromarray((image_np * 255).astype(np.uint8))
    img_ooo = Image.fromarray((image_np * 255).astype(np.uint8))

    draw = ImageDraw.Draw(img_pil)

    draw_gt = ImageDraw.Draw(img_ori)

    font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', size=28)
    # font=ImageFont.load_default()
    # font_size = 36

    colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 紫色
        (0, 255, 255),    # 青色
        (255, 255, 255),  # 白色
        (255, 165, 0),    # 橙色
        (0, 0, 128),      # 深蓝色
        (165, 42, 42)     # 棕色
    ]

    def draw_dashed_rect(draw, bbox, color):
        # 绘制虚线矩形
        for i in range(0, int(bbox[2] - bbox[0]), 10):
            draw.line([(bbox[0] + i, bbox[1]), (bbox[0] + i + 5, bbox[1])], fill=color, width=2)
            draw.line([(bbox[0] + i, bbox[3]), (bbox[0] + i + 5, bbox[3])], fill=color, width=2)
        for i in range(0, int(bbox[3] - bbox[1]), 10):
            draw.line([(bbox[0], bbox[1] + i), (bbox[0], bbox[1] + i + 5)], fill=color, width=2)
            draw.line([(bbox[2], bbox[1] + i), (bbox[2], bbox[1] + i + 5)], fill=color, width=2)

    def mapping(parts):
        if parts == '5':
            out = 'SP'
        elif parts == '1':
            out = 'RA'
        elif parts == '3':
            out = 'LV'
        elif parts == '6':
            out = 'LA'
        elif parts == '2':
            out = 'RV'
        elif parts == '4':
            out = 'VS'
        elif parts == '7':
            out = 'CR'
        elif parts == '8':
            out = 'DAO'
        elif parts == '9':
            out = 'R'
        # else:
        #     out = 'ROI'
        return out

    # 在图像上绘制边界框和标签
    for label in np.unique(gt_labels):
        # 获取ground truth中相同标签的个数
        num_gt = np.sum(gt_labels == label)

        # 获取该标签的所有预测框，并按照得分进行排序
        mask = pred_labels == label
        sorted_indices = np.argsort(pred_scores[mask])[::-1][:num_gt]

        # 画出预测的边界框
        for idx in sorted_indices:
            bbox = pred_bboxes[mask][idx]
            score = pred_scores[mask][idx]
            color = colors[label]
            draw.rectangle(bbox, outline=color, width=4)
            draw.text((bbox[0]+10, bbox[1]), f"{mapping(str(label))}", font=ImageFont.load_default(), fill=color)
        if gt:
        # 画出Ground Truth的边界框（虚线）
            for bbox in gt_bboxes[gt_labels == label]:
                draw_gt.rectangle(bbox, outline=color, width=4)
                draw_gt.text((bbox[0]+5, bbox[1]+5), f"{mapping(str(label))}", font=font, fill=color)
                # draw_dashed_rect(draw, bbox, color)

    # 保存图像
    # timestr = time.strftime('%m%d%H%M%S')
    img_pil.save(f'{save_path}/{ids}_bboxes.png')
    img_ooo.save(f'{save_path}/{ids}_orig.png')
    if gt:
        img_ori.save(f'{save_path}/{ids}_gt.png')
    # img_pil.show()
