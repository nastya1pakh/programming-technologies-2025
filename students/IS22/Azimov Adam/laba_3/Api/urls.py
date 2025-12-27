from django.urls import path
from . import views
from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

urlpatterns = [
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/swagger/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
    path("collections/", views.get_list_collections, name="get_list_collections"),
    path("collections/add/", views.add_collections, name="add_collections"),
    path("collections/<str:collection_name>/", views.delete_collections, name="delete_collections"),
    path("collections/info/<str:collection_name>/", views.get_collection_info, name="get_collection_info"),
    path("text/add/", views.add_text, name="add_text"),
    path("search/", views.search_query, name="search_query"),
]
