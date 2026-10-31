# Codex 사용량 위젯

Windows 바탕화면에 Codex 사용량을 항상 표시하는 작은 위젯입니다.

## 사용 방법

1. `Codex 사용량.exe`를 더블클릭합니다.
2. Codex 앱에 ChatGPT 계정으로 로그인되어 있어야 합니다.
3. 10초마다 5시간 및 주간 사용량을 새로 읽습니다.

투명도는 아래의 원형 손잡이를 좌우로 드래그해 10~100% 범위에서 조절합니다. 투명도 손잡이를 제외한 제목이나 사용량 영역을 드래그하면 위젯 창을 옮길 수 있습니다.

## 소스에서 실행

```powershell
python codex_usage_widget.py
```

Python 실행에는 Tkinter가 필요하며, Codex CLI가 설치되어 있어야 합니다. 위젯은 Codex app-server의 `account/rateLimits/read` 요청을 사용해 현재 계정의 사용량과 초기화 시각을 읽습니다.

## 구성

- `Codex 사용량.exe`: Python 설치 없이 실행할 수 있는 Windows 실행 파일
- `codex_usage_widget.py`: 위젯 소스

설정 파일은 `%LOCALAPPDATA%\CodexUsageWidget\settings.json`에 저장됩니다.
