signalavergecg
📦 Установка и подготовка
Установите Python 3.11+:

python.org/downloads
Или через pyenv
➤ Для Windows: pyenv-win
Склонируйте проект:

git clone https://github.com//signalavergecg
cd signalavergecg
Установите uv:
pip install uv
Создайте виртуальное окружение и активируйте его:
uv venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
Синхронизация зависимостей в активном окружении
uv sync --active
