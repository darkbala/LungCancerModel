from django.db import models

class LungScan(models.Model):
    image = models.ImageField(upload_to='scans/')
    result = models.CharField(max_length=100, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan {self.id} - {self.result}"
