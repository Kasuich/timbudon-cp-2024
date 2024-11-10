.PHONY: docker_up
docker_up:
	sudo docker compose up --build

.PHONY: docker_up_d
docker_up_d:
	sudo docker compose up --build -d

.PHONY: docker_down
docker_down:
	sudo docker compose down

.PHONY: update_proto
update_proto:
	python -m grpc_tools.protoc -Iprotos --python_out=ml_predictor/pb --pyi_out=ml_predictor/pb --grpc_python_out=ml_predictor/pb protos/inference.proto
	python -m grpc_tools.protoc -Iprotos --python_out=backend/pb --pyi_out=backend/pb --grpc_python_out=backend/pb protos/inference.proto