from django.contrib import admin
from django.urls import path
from django.http import JsonResponse
from api.ninja_api import api_v1


def root_view(request):
    return JsonResponse({"message": "FertiNex API is running!"})


def health_view(request):
    return JsonResponse({"status": "healthy"})


urlpatterns = [
    path("", root_view, name="root"),
    path("health", health_view, name="health"),
    path("health/", health_view, name="health_slash"),
    path("api/v1/", api_v1.urls),
    path("django-admin/", admin.site.urls),
]
