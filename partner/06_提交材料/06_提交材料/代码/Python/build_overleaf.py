# -*- coding: utf-8 -*-
"""生成 Overleaf 单文件版论文: 08_Overleaf/main.tex (全部章节内联, 图片同级目录)
   · 不依赖 cumcmthesis.cls / cumcm2026.sty, 改用 ctexart + 常用宏包的自足导言区
   · 图片复制到 08_Overleaf/ 并与 main.tex 同级引用
"""
import sys, io, os, re, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, '05_论文与结果', '论文')
OUT = os.path.join(ROOT, '08_Overleaf')
os.makedirs(OUT, exist_ok=True)

sections = ['sec1_restate', 'sec2_analysis', 'sec3_patterns', 'sec4_assume', 'sec5_q1',
            'sec6_q2', 'sec7_q3', 'sec8_q4', 'sec9_compare', 'sec10_eval']
main = open(os.path.join(SRC, 'main.tex'), encoding='utf-8').read()

# ---------- 1) 取出摘要正文 ----------
ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main, re.S).group(1)
ab = ab.replace('\\keywords{', '\\cumcmkeywords{')
# 去掉摘要中的空行(保持段落)
ab = re.sub(r'\n\s*\n', '\n', ab).strip()

# ---------- 2) 取出参考文献 ----------
bib = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', main, re.S).group(0)

# ---------- 3) 拼接章节 ----------
def body(name):
    t = open(os.path.join(SRC, 'sections', name + '.tex'), encoding='utf-8').read()
    return t.replace('figures/', '')

chunks = [body(s) for s in sections]
apx = body('appendix')
# 附录中的 subsection 层级保持不变, 由 \appendix 自动改为 A/B/C
apx = apx.replace('\\section{程序与数据清单}', '\\section{程序与数据清单}')

preamble = r"""% !TEX program = xelatex
% 2026 高教社杯全国大学生数学建模竞赛 C 题
% 微网与外部电网电力调控策略 —— Overleaf 单文件版(全部章节内联, 图片与 main.tex 同目录)
% 编译方式: XeLaTeX(Overleaf 默认即可), 无需任何本地字体/模板文件
\documentclass[12pt,a4paper,fontset=fandol]{ctexart}

\usepackage{geometry}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

% 西文字体改为 Times 系(TeX Gyre Termes)/ Helvetica 系: 与国赛模板的西文字体一致,
% 且数字更窄, 三线表放大字号后仍能落在版心内(TeX Live 与 Overleaf 均自带, 无需配置)
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}

% 图片搜索路径: 见下方 graphicx 之后(必须在 graphicx 之后设置才生效)
\usepackage{amsmath,amssymb,mathtools}
\usepackage{bm}
\usepackage{booktabs,array,tabularx,longtable,multirow}
\usepackage{graphicx,float,caption,subcaption}

% 图片搜索路径(必须在 graphicx 之后设置, 否则 \graphicspath 未定义会报错失效):
% 既支持图片与 main.tex 同目录, 也支持统一放在 figures\ 子目录
\graphicspath{{figures/}{./}}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{url}
\usepackage{tcolorbox}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usepackage{fancyhdr}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!55!black,citecolor=green!45!black,urlcolor=blue!60!black}
\usepackage{cleveref}

\newcolumntype{Y}{>{\centering\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}

\definecolor{cumcmblue}{HTML}{1F4E79}
\definecolor{cumcmgray}{HTML}{F2F4F7}
\definecolor{codegreen}{HTML}{2E6B4E}

\setlength{\parindent}{2em}
\linespread{1.35}
\setlist{leftmargin=2em,labelsep=0.6em,itemsep=0.25em,topsep=0.4em,parsep=0pt}
\captionsetup{font=small,labelsep=quad,skip=6pt}

% 行内少量伸缩: 避免文字越出右边界, 也避免句末标点被挤到单独一行
\setlength{\emergencystretch}{3em}
% 三线表: 略微收紧列间距与行距, 以便在有限宽度内用更大的字号
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}

% 圈码 ①—⑳ 等符号归入中文字体类(西文字体无此字形)
\xeCJKDeclareCharClass{CJK}{"2460 -> "24FF, "3001 -> "303F, "FF01 -> "FF5E}

% cleveref 中文名称
\crefname{figure}{图}{图}
\crefname{table}{表}{表}
\crefname{section}{节}{节}
\crefname{equation}{式}{式}
\creflabelformat{equation}{(#2#1#3)}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}

% 代码样式
\lstset{
  basicstyle=\ttfamily\scriptsize,
  breaklines=true,
  breakatwhitespace=false,
  columns=flexible,
  frame=single,
  framexleftmargin=2pt,
  numbers=left,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{cumcmblue}\bfseries,
  commentstyle=\color{codegreen},
  stringstyle=\color{orange!70!black},
  showstringspaces=false,
  captionpos=b,
  extendedchars=true,
  tabsize=2
}
\renewcommand{\lstlistingname}{代码}

\newcommand{\cumcmkeywords}[1]{\par\vspace{0.6em}\noindent{\heiti\bfseries 关键词：}#1}
\newcommand{\CumcmAIUsed}[1]{%
  \section*{AI 工具使用声明}
  本参赛队在竞赛过程中使用了 AI 工具，主要用于 #1，详细使用情况见支撑材料。}
"""

doc = []
doc.append(preamble)
doc.append(r'\begin{document}' + '\n')
doc.append(r'\begin{center}{\zihao{2}\heiti 微网与外部电网电力调控策略}\end{center}' + '\n')
doc.append(r'\vspace{0.5em}' + '\n')
doc.append(r'\begin{center}{\zihao{4}\heiti 摘\quad 要}\end{center}' + '\n')
doc.append(ab + '\n')
doc.append(r'\newpage' + '\n')
for c in chunks:
    doc.append(c.strip() + '\n\n')
doc.append(r'\CumcmAIUsed{代码实现与调试、文献资料整理、图表绘制脚本编写与文字润色}' + '\n\n')
doc.append(bib + '\n\n')
doc.append(r'\appendix' + '\n')
doc.append(apx.strip() + '\n')
doc.append(r'\end{document}' + '\n')

open(os.path.join(OUT, 'main.tex'), 'w', encoding='utf-8').write('\n'.join(doc))

# ---------- 4) 复制图片 ----------
figs = set(re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', '\n'.join(doc)))
for f in sorted(figs):
    src = os.path.join(SRC, 'figures', f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(OUT, os.path.basename(f)))
    else:
        print('  [缺失]', f)
print('图片 %d 个 -> %s' % (len(figs), OUT))
print('main.tex 行数:', len(open(os.path.join(OUT, 'main.tex'), encoding='utf-8').readlines()))
