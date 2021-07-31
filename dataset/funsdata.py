import logging
logger = logging.getLogger()
import os
import json
from data_util import load_image,normalize_bbox

def _generate_examples(filepath):
    logger.info("⏳ Generating examples from = %s", filepath)
    ann_dir = os.path.join(filepath, "annotations")
    img_dir = os.path.join(filepath, "images")

    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        tokens = []
        bboxes = []
        ner_tags = []

        file_path = os.path.join(ann_dir, file)
        _, file_name_stem = os.path.split(file_path)
        fname, _ = os.path.splitext(file_name_stem)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        each_file_res = []
        for item in data["form"]:
            tmp_tokens = []
            tmp_bboxes = []
            tmp_ner_tags = []
            each_file_dict = dict()
            words, label = item["words"], item["label"]
            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            if label == "other":
                for w in words:
                    tokens.append(w["text"])
                    tmp_tokens.append(w["text"])
                    ner_tags.append("O")
                    tmp_ner_tags.append("O")
                    bboxes.append(normalize_bbox(w["box"], size))
                    tmp_bboxes.append(normalize_bbox(w["box"], size))
            else:
                tokens.append(words[0]["text"])
                tmp_tokens.append(words[0]["text"])
                ner_tags.append("B-" + label.upper())
                tmp_ner_tags.append("B-" + label.upper())
                bboxes.append(normalize_bbox(words[0]["box"], size))
                tmp_bboxes.append(normalize_bbox(words[0]["box"], size))
                for w in words[1:]:
                    tokens.append(w["text"])
                    tmp_tokens.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    tmp_ner_tags.append("I-" + label.upper())
                    bboxes.append(normalize_bbox(w["box"], size))
                    tmp_bboxes.append(normalize_bbox(w["box"], size))

            each_file_dict["tokens"]=tmp_tokens
            each_file_dict['ner_tags']=tmp_ner_tags
            each_file_dict["bbox"] = tmp_bboxes
            each_file_res.append(each_file_dict)
        with open("C:/Users/QIANGHAO/Desktop/CV_Code/MyLM/training_data/{}".format(fname)+".json",'w',encoding='utf-8') as fw:
            json.dump(each_file_res,fw)
            # with open("/mnt/data/competition/pythonProject/pythonProject/layoutlmv2/funds_data/extracted_data/{}".format(fname)+".txt",'a',encoding='utf-8') as fw:
            #     fw.write(str(tmp_tokens+tmp_bboxes+tmp_ner_tags))

if __name__=="__main__":
    filepath="C:/Users/QIANGHAO/Desktop/CV_Code/funsd/dataset/training_data"
    _generate_examples(filepath)

