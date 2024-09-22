
build : 
	docker build -t titanicapp .

PORT ?= 8000
run : 
	docker run -d -p 8000:$(PORT) -e PORT=$(PORT) titanicapp

build_gcr :
	docker build --platform linux/amd64 -t gcr.io/titanicsurvival-436410/my-fast-api-app:latest .

push_gcr :
	docker push gcr.io/titanicsurvival-436410/my-fast-api-app:latest

deploy_gcr :
	gcloud run deploy titanic-survival-service \
  --image gcr.io/titanicsurvival-436410/my-fast-api-app:latest \
  --platform managed \
  --region europe-west4 \
  --allow-unauthenticated 