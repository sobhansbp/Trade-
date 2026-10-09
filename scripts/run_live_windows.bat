@echo off
REM Live demo run on Windows with MetaTrader 5.
REM 1) Install Python 3.12 and run:  pip install -r requirements.txt MetaTrader5
REM 2) Open MetaTrader 5, log in to a DEMO account, enable Tools > Options > Expert Advisors > "Allow algorithmic trading"
REM 3) Set your keys below (never commit them), then double-click this file.
REM    If your broker uses other symbol names, set e.g. MT5_SYMBOL_XAUUSD=XAUUSDm
cd /d %~dp0\..
set GROQ_API_KEY=
set TELEGRAM_BOT_TOKEN=
set TELEGRAM_CHAT_ID=
set MT5_SYMBOL_EURUSD=EURUSD
set MT5_SYMBOL_XAUUSD=XAUUSD
REM Paper only (no orders):
REM python -m aitrader live EURUSD XAUUSD --ai
REM Orders to the logged-in MT5 account (use a DEMO account!):
python -m aitrader live EURUSD XAUUSD --ai --execute
pause
