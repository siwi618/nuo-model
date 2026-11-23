"""把人工核对的题目（items.json）填进朋友版单文件测验 scripts/human_check_quiz.html（2026-11-23）。

用法（仓库根目录）：.venv/bin/python scripts/human_check_quiz_build.py data/human-check/items.json data/human-check/台词耳朵测试.html
生成的 html 不联网也能用，微信 / 邮件发给朋友，浏览器打开即可；做完给一行结果码（台词耳朵测试 | 名字 | NUO1…），
把它存成 txt 交给 human_check_score.py 就能算。题目顺序和 items.json 一样（单评 s01–s40，再两两 p01–p20），解码靠这个顺序。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    items = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    order = [i for i in items if i["kind"] == "single"] + [i for i in items if i["kind"] == "pair"]
    assert [i["id"] for i in order] == [i["id"] for i in items], "items.json 要先单评后两两"
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    html = (HERE / "human_check_quiz.html").read_text(encoding="utf-8").replace("/*ITEMS*/", data)
    Path(sys.argv[2]).write_text(html, encoding="utf-8")
    print(f"{len(items)} 题 → {sys.argv[2]}（{len(html.encode()) // 1024} KB）")


if __name__ == "__main__":
    main()
