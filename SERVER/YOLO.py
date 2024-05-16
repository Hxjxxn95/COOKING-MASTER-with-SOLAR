from ultralytics import YOLO
from PIL import Image
import io
import cv2
import base64
import numpy as np

class YOLO:
    model = YOLO('./YOLOv8/best.pt')
    def __init__(self):
        self.yolo = self.model
        self.yolo.conf = 0.3

    def convert_image(self, input):
        if not isinstance(input, (bytes, bytearray)):
            raise TypeError("Expected bytes-like object, got %s" % type(input).__name__)
        image_bytes = io.BytesIO(input)
        image = Image.open(image_bytes)
        return image

    def __call__(self, input):
        input = self.convert_image(input)
        results = self.yolo(input, stream=True)
        class_names = self.model.names
        self.objects_detected = set()
        
        img = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
    
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            names = result.names
            confidences = result.boxes.conf.tolist()
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = conf
                detected_class = cls
                name = names[int(cls)]
                self.objects_detected.add(name)
                
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(img, str(round(confidence, 2)), (int(x1), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        name_change  ={
            'chicken_breast': '닭가슴살',
            'goat_cheese': '염소치즈',
            'ground_beef': '다진소고기',
            'eavy_cream': '생크림',
            'sweet_potato': '고구마',
        }
        # Change the name of the detected object to Korean
        self.true_name_detected = [name_change.get(name, name) for name in self.objects_detected]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # transform the image to PIL format
        img_pil = Image.fromarray(img)

        # save the image to a byte array
        byte_arr = io.BytesIO()
        img_pil.save(byte_arr, format='PNG')
        encoded_image = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        
        return {'result':self.true_name_detected, 'image':encoded_image}
    