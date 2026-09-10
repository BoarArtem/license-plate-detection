from ultralytics import YOLO

def train(model_name, data_path):
    model = YOLO(model_name)

    model.train(
        data=data_path,
        epochs=150,
        imgsz=640,
    )


if __name__ == '__main__':
    train("yolo26l.pt", "../data/data.yaml")