# Generated by Django 5.0 on 2024-03-15 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Image_Path",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=200)),
                ("path", models.CharField(max_length=200)),
                ("uploaded_date", models.DateTimeField(verbose_name="date uploaded")),
            ],
        ),
    ]
