# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mlapi.urls')),  # 将 mlapi 的 URL 路由包含进来
]
