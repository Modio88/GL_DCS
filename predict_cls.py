from predict_multibans import MultiBandClassificationPredictor
def classify_gl(input,out_classify):
    args = dict(
        model="classify.pt",
        source=input,
        project=out_classify,
        imgsz=224,
        save=False
    )

    predictor = MultiBandClassificationPredictor(overrides=args)
    predictor.predict_cli()
    predictor.save_excel()