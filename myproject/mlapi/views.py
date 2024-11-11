# mlapi/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .models import UploadedImage
from .model import predict_image
from PIL import Image

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({"error": "No image provided"}, status=400)

        try:
            image = Image.open(image_file)
            image.verify()  # 验证图像文件
            image_file.seek(0)  # 重置文件指针
        except Exception as e:
            return JsonResponse({"error": f"Invalid image file: {e}"}, status=400)

        uploaded_image = UploadedImage(image=image_file)
        uploaded_image.save()

        try:
            image_file.seek(0)  # 再次读取文件
            prediction = predict_image(image_file.read())
        except Exception as e:
            return JsonResponse({"error": f"Error during prediction: {e}"}, status=500)

        uploaded_image.prediction = prediction
        uploaded_image.save()

        return JsonResponse({"class": prediction})  # 确保返回 JSON 响应

    return JsonResponse({"error": "Invalid request method"}, status=405)

def index(request):
    return render(request, 'index.html')  # 返回主页模板
