from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from . import nermodel

# Create your views here.

def index(request): 
    return HttpResponse("Hello")

def get_ner_tag(request):
    sentence = request.GET.get('sentence',None)
    # return HttpResponse(f"Hello {sentence}")
    tag_result = nermodel.string2tag(sentence)
    data = {
    'summary': tag_result,
    'raw': 'Successful',
    }
    return JsonResponse(data)
