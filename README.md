# DALLEePaperFrame

## Getting Started
### Setup the Server

Setup the server with the script `setup_server.sh`:
```bash
cd server/
bash setup_server.sh <your_wandb_api_token> # the API token you get from https://wandb.com/authorize
```

### Run the Server
```bash
cd server/
bash run_server.sh <ip_address> # the IP address of the server
```

### Setup the Client
2. Setup the client with the script `setup_client.sh`:
```bash
cd client/
bash setup_client.sh 
```

### Run the Client
```bash
cd client/
bash run_client.sh <ip_address> # the IP address of the server
```

### Reducing power consumption
https://www.cnx-software.com/2021/12/09/raspberry-pi-zero-2-w-power-consumption/
https://blues.io/blog/tips-tricks-optimizing-raspberry-pi-power/