from django import forms
from .models import LungScan

class LungScanForm(forms.ModelForm):
    class Meta:
        model = LungScan
        fields = ('image',)
