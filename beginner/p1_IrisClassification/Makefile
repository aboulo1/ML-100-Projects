
build : 
	docker build -t irisapp .

PORT ?= 8000
run : 
	docker run -d -p 8000:$(PORT) -e PORT=$(PORT) irisapp

buildgcr :
	docker build --platform linux/amd64 -t gcr.io/irisclassifier-432714/my-fast-api-app:latest .



push_gcr :
	docker push gcr.io/irisclassifier-432714/my-fast-api-app:latest

deploy_gcr :
	gcloud run deploy iris-classification-service \
  --image gcr.io/irisclassifier-432714/my-fast-api-app:latest \
  --platform managed \
  --region europe-west4 \
  --allow-unauthenticated 
