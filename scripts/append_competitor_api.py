import os

# APIに追加するコード (Competitor Graph)
code_to_append = """

# ── 競合関係グラフ
@app.get("/api/analysis/competitor_graph")
def api_competitor_graph():
    from components.graph_view import build_graph_data
    try:
        data = build_graph_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api', 'main.py'), 'a') as f:
    f.write(code_to_append)
print("Appended Competitor Graph API to main.py")
