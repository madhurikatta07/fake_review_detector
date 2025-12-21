from features import simple_text_features

text = "Amazing product!!! 😍 Worth the price. Visit http://fake.com"
features = simple_text_features(text, ocr_conf=0.93)

for k, v in features.items():
    print(k, ":", v)
