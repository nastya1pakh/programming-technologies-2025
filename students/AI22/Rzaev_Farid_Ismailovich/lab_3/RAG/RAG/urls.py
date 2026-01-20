from django.contrib import admin
from django.urls import include, path
from .controllers.milvus import MilvusController
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from rest_framework.routers import DefaultRouter

schema_view = get_schema_view(
    openapi.Info(
        title="API",
        default_version='v1',
        description="Описание API",
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)
router = DefaultRouter()
router.register(r'milvus', MilvusController, basename="milvus")


urlpatterns = [

    path('docs/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path("admin/", admin.site.urls),
    path('', include(router.urls))
]
