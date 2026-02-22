"""
リース審査 Web アプリ（Flask）
モダンなUIで簡易審査スコアを表示。詳細は Streamlit 版を利用。
"""
import os
import sys
import json

# リポジトリルートと lease_logic_sumaho10 をパスに追加（web/app.py から見て ../.. = lease_logic_sumaho10, ../../.. = リポジトリルート）
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMODULE = os.path.dirname(_WEB_DIR)   # lease_logic_sumaho10
_REPO_ROOT = os.path.dirname(_SUBMODULE)  # リポジトリルート
for p in (_REPO_ROOT, _SUBMODULE, _WEB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-in-production")
# 403 を防ぐ: 127.0.0.1 / localhost を明示的に許可（Host ヘッダ検証）
app.config["TRUSTED_HOSTS"] = [
    "127.0.0.1",
    "localhost",
    "127.0.0.1:5000",
    "localhost:5000",
]


def _allow_localhost_wsgi(app_wrapper):
    """Host が 127.0.0.1 / localhost / LAN(192.168.x.x) いずれでも受け付ける。403 防止のため localhost に正規化。"""
    def wsgi(environ, start_response):
        host = environ.get("HTTP_HOST", "").split(":")[0]
        port = environ.get("SERVER_PORT", "5050")
        if host in ("127.0.0.1", "localhost", "::1"):
            environ["HTTP_HOST"] = f"localhost:{port}"
        elif host.startswith("192.168.") or host.startswith("10."):
            environ["HTTP_HOST"] = f"{host}:{port}"
        return app_wrapper(environ, start_response)
    return wsgi


app.wsgi_app = _allow_localhost_wsgi(app.wsgi_app)


@app.route("/health")
def health():
    """接続確認用。スマホや他端末から 192.168.x.x:5050/health で応答があればサーバーは起動している。"""
    return jsonify({"status": "ok", "service": "lease-web"}), 200


def _load_benchmarks():
    path = os.path.join(_REPO_ROOT, "industry_benchmarks.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _sub_to_major(industry_sub: str) -> str:
    """業種中分類コードから大分類を推測。"""
    if not industry_sub:
        return "D 建設業"
    code = industry_sub.split()[0] if industry_sub else ""
    if code.startswith("0"):
        return "D 建設業"
    if code.startswith("0") or code in ("09", "21", "24", "26"):
        return "E 製造業"
    if code in ("43", "44"):
        return "H 運輸業"
    if code in ("50", "56") or "卸売" in industry_sub or "小売" in industry_sub:
        return "I 卸売業・小売業"
    if code in ("68", "70", "75", "76"):
        return "M その他サービス業"
    if code == "83":
        return "P 医療・福祉"
    return "D 建設業"


@app.route("/")
def index():
    benchmarks = _load_benchmarks()
    industry_subs = sorted(benchmarks.keys()) if benchmarks else ["06 総合工事業"]
    return render_template("index.html", industry_subs=industry_subs)


@app.route("/result", methods=["POST"])
def result():
    try:
        from scoring_core import run_quick_scoring
    except ImportError:
        return render_template("error.html", message="スコア計算モジュールの読み込みに失敗しました。"), 500

    industry_sub = request.form.get("industry_sub", "06 総合工事業").strip()
    industry_major = request.form.get("industry_major", _sub_to_major(industry_sub)).strip()

    def _float(key, default=0):
        try:
            return float(request.form.get(key) or default)
        except (TypeError, ValueError):
            return float(default)

    def _int(key, default=0):
        try:
            return int(request.form.get(key) or default)
        except (TypeError, ValueError):
            return int(default)

    # フォームは千円単位・1千円刻みで送られる（そのままスコア計算に渡す）
    inputs = {
        "nenshu": _float("nenshu"),
        "op_profit": _float("op_profit"),
        "rieki": _float("op_profit"),
        "ord_profit": _float("ord_profit"),
        "net_income": _float("net_income"),
        "net_assets": _float("net_assets"),
        "total_assets": _float("total_assets"),
        "industry_major": industry_major,
        "industry_sub": industry_sub,
        "grade": request.form.get("grade", "1-3"),
        "customer_type": request.form.get("customer_type", "既存先"),
        "bank_credit": _float("bank_credit"),
        "lease_credit": _float("lease_credit"),
        "contracts": _int("contracts"),
        "gross_profit": _float("gross_profit"),
        "machines": _float("machines"),
        "other_assets": _float("other_assets"),
        "rent": _float("rent"),
        "depreciation": _float("depreciation"),
        "dep_expense": _float("dep_expense"),
        "rent_expense": _float("rent_expense"),
    }

    if inputs["total_assets"] <= 0 or inputs["nenshu"] <= 0:
        flash("売上高と総資産は 1 千円以上を入力してください。", "error")
        return redirect(url_for("index"))

    res = run_quick_scoring(inputs)
    return render_template("result.html", result=res)


@app.route("/error")
def error():
    return render_template("error.html", message=request.args.get("message", "エラーが発生しました。"))


@app.route("/visualization")
def visualization():
    """スコア可視化ページ（データビュー + Tune Space）。"""
    return render_template("visualization.html")


@app.route("/api/visualization/data")
def api_visualization_data():
    """可視化用JSONを返す。past_cases ベース＋不足分はダミー。"""
    try:
        from export_visualization_data import get_visualization_data
        return jsonify(get_visualization_data())
    except Exception as e:
        return jsonify({"error": str(e), "data": [], "total_count": 0, "generated_at": None}), 500


if __name__ == "__main__":
    import os as _os
    import socket as _socket
    port = int(_os.environ.get("PORT", 5050))
    url_local = f"http://127.0.0.1:{port}"
    url_localhost = f"http://localhost:{port}"
    lan_url = None
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        lan_url = f"http://{lan_ip}:{port}"
    except Exception:
        pass
    print(f" * 簡易審査・可視化: {url_local} または {url_localhost}")
    if lan_url:
        print(f" * スマホ／他PCから: {lan_url}  （同じWi-Fi内）")
    print(f" * 接続確認: {lan_url or url_local}/health")
    use_flask_dev = _os.environ.get("FLASK_DEV_SERVER", "").lower() in ("1", "true", "yes")
    if use_flask_dev:
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        try:
            from waitress import serve
            serve(app, host="0.0.0.0", port=port)
        except ImportError:
            from wsgiref.simple_server import make_server
            with make_server("", port, app) as httpd:
                httpd.serve_forever()
