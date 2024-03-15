from django.apps import AppConfig
from ultralytics import YOLO
from collections import Counter


class RecognizerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "recognizer"

    
class ImagePredictor:
    def __init__(self, image_path, task):
        self.image_path = image_path
        self.task = task

    def refactor_results(self, results):
        output_list = []
        for result in results:
            list_of_predicted_fruits = []
            dictionary_of_classes = result.names
            for i in result.boxes.data.cpu().numpy():
                list_of_predicted_fruits.append(dictionary_of_classes[int(i[-1])])

            fruit_counts = Counter(list_of_predicted_fruits)
            output = ", ".join([f"{count} {fruit if count == 1 else fruit + 's'}" for fruit, count in fruit_counts.items()])
            output_list.append(output)
        return output_list
    
    def predict(self):
        model = YOLO("recognizer/best.onnx", task=self.task)
        results = model(self.image_path)
        predictions = self.refactor_results(results)
        return predictions