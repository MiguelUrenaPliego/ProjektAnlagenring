
# 📊 ABsurveys — Full Deployment & Setup Guide

ABsurveys is a full-stack survey platform with:

* ⚡ FastAPI backend
* 🧠 MongoDB (TrueSkill-based ranking system)
* 🌐 React frontend (served via FastAPI)
* 🚀 Nginx reverse proxy
* 🔐 Optional HTTPS via Certbot

---

# 📦 Project Structure

```
ABsurveys/
├── config.json
├── backend/
│   └── server.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── index.css
│       ├── App.jsx
│       ├── App.css
│       └── components/
├── images/
│   └── <ScenarioName>/
│       ├── images.csv
│       ├── 1.jpg
│       └── ...
├── questions/
├── scenarios.csv
└── user_data/
    └── mongodb/
```

---

# 🧰 Prerequisites

### Server

* Debian 12/13 or Ubuntu 22+
* SSH access with sudo privileges

### Local machine

* Node.js 18+
* npm
* rsync

---

# 🏗️ 1. Build Frontend Locally

```bash
cd ABsurveys/frontend
npm install
npm run build
```

Verify output:

```bash
ls frontend/dist/
# index.html assets/
```

---

# 🚀 2. Upload Project to Server

From project root:

```bash
rsync -av \
  --exclude='user_data/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='frontend/node_modules/' \
  ./ user@your-server.com:/home/user/ABsurveys/
```

Upload frontend build:

```bash
rsync -av frontend/dist/ \
  user@your-server.com:/home/user/ABsurveys/frontend/dist/
```

---

# 🧱 3. Install MongoDB 8.0

```bash
sudo apt update
sudo apt install -y curl gnupg

curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc \
  | sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] \
https://repo.mongodb.org/apt/debian bookworm/mongodb-org/8.0 main" \
  | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt update
sudo apt install -y mongodb-org

sudo systemctl start mongod
sudo systemctl enable mongod
```

---

# 💾 4. Move MongoDB into Project Directory

```bash
sudo systemctl stop mongod

mkdir -p /home/user/ABsurveys/user_data/mongodb

sudo rsync -av /var/lib/mongodb/ /home/user/ABsurveys/user_data/mongodb/

sudo chown -R mongodb:mongodb /home/user/ABsurveys/user_data/mongodb
```

Bind mount:

```bash
sudo mount --bind \
  /home/user/ABsurveys/user_data/mongodb \
  /var/lib/mongodb
```

Persist:

```bash
echo "/home/user/ABsurveys/user_data/mongodb /var/lib/mongodb none bind 0 0" \
  | sudo tee -a /etc/fstab
```

Restart:

```bash
sudo systemctl start mongod
```

---

# 🐍 5. Install Python Dependencies (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd /home/user/ABsurveys
uv add fastapi uvicorn pandas numpy motor trueskill
```

---

# 🧪 6. Test Backend Manually

```bash
cd /home/user/ABsurveys

export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="ABsurveys"

uv run uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

Verify output:

```
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

# ⚙️ 7. Create systemd Service

```bash
sudo nano /etc/systemd/system/absurveys.service
```

Paste:

```ini
[Unit]
Description=ABsurveys
After=network.target mongod.service
Requires=mongod.service

[Service]
User=user
WorkingDirectory=/home/user/ABsurveys
Environment="MONGODB_URI=mongodb://localhost:27017"
Environment="MONGODB_DB=ABsurveys"
Environment="PATH=/home/user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/user/.local/bin/uv run uvicorn backend.server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable absurveys
sudo systemctl start absurveys
```

---

# 🌐 8. Configure Nginx

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/default
```

Replace:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

# 🔐 9. HTTPS (Optional)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-server.com
```

---

# 🔄 Updating the App

## Rebuild frontend locally

```bash
cd frontend
npm run build
```

## Upload changes

```bash
rsync -av frontend/dist/ user@your-server.com:/home/user/ABsurveys/frontend/dist/
rsync -av backend/ user@your-server.com:/home/user/ABsurveys/backend/
rsync -av config.json user@your-server.com:/home/user/ABsurveys/
```

## Restart backend

```bash
ssh user@your-server.com "sudo systemctl restart absurveys"
```

---

# 📊 Inspecting Data

```bash
mongosh absurveys
show collections
db.user_answers.countDocuments()
db.image_state.countDocuments()
```

---

# 📥 Download Survey Results

## User answers

```bash
mongoexport --db=absurveys --collection=user_answers --type=csv \
  --fields=user_id,scenario,language,question_id,type,answer,img_id_A,img_id_B,img_type,info \
  --out=/home/user/ABsurveys/user_data/user_answers.csv
```

```bash
rsync -av user@your-server.com:/home/user/ABsurveys/user_data/user_answers.csv ./user_data/
```

---

## Image state (TrueSkill)

```bash
mongoexport --db=absurveys --collection=image_state \
  --type=json --jsonArray \
  --out=/home/user/ABsurveys/user_data/image_state.json
```

```bash
rsync -av user@your-server.com:/home/user/ABsurveys/user_data/image_state.json ./user_data/
```

Convert locally:

```python
import pandas as pd

df = pd.read_json("user_data/image_state.json")
df.to_csv("user_data/image_state.csv", index=False)
```

---

# 📤 Upload New Data

## New images

```bash
rsync -av images/NewScenario/ \
  user@your-server.com:/home/user/ABsurveys/images/NewScenario/
```

---

## New config.json

```bash
rsync -av config.json \
  user@your-server.com:/home/user/ABsurveys/config.json
```

---

## New scenarios.csv

```bash
rsync -av scenarios.csv \
  user@your-server.com:/home/user/ABsurveys/scenarios.csv
```

---

## Restart after updates

```bash
ssh user@your-server.com "sudo systemctl restart absurveys"
```

---

# 🔁 Scenario Reset

```bash
mongosh absurveys --eval "db.image_state.drop()"
sudo systemctl restart absurveys
```

---

# 🧨 Wipe All Data (RESET)

⚠️ irreversible

```bash
mongosh absurveys --eval "db.image_state.drop(); db.user_answers.drop()"
sudo systemctl restart absurveys
```

---

# 📜 Logs

```bash
journalctl -u absurveys.service -f
journalctl -u absurveys.service -n 50 --no-pager
journalctl -u mongod -n 50 --no-pager
```

---

# 📁 Production Paths

| Component    | Path                                      |
| ------------ | ----------------------------------------- |
| App          | `/home/user/ABsurveys/`                   |
| MongoDB data | `/home/user/ABsurveys/user_data/mongodb/` |
| Config       | `/home/user/ABsurveys/config.json`        |
| Scenarios    | `/home/user/ABsurveys/scenarios.csv`      |
| Images       | `/home/user/ABsurveys/images/<Scenario>/` |
| Service      | `/etc/systemd/system/absurveys.service`   |
| Nginx        | `/etc/nginx/sites-available/default`      |
