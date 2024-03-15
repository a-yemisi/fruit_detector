from django.shortcuts import render, redirect
from django.http import HttpResponse
from .apps import ImagePredictor
from .forms import ImageUploadForm

# Create your views here.
def upload_image(request):
    results = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            image_path = uploaded_image.image.path
            image_predictor = ImagePredictor(image_path, task="detect")
            results = image_predictor.predict()
    else:
        form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form, "results":results})
