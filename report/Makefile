# 编译技术报告为 PDF（需 XeLaTeX + ctex 中文支持）
# 用法: make       生成 report.pdf
#       make clean 清理中间文件

TEX = report.tex
PDF = report.pdf

all: $(PDF)

$(PDF): $(TEX)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz

.PHONY: all clean
