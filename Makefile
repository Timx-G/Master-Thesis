PYTHON ?= python
TECTONIC ?= tectonic
CANONICAL ?= artifacts/canonical/v1

.PHONY: test assets paper thesis verify

test:
	$(PYTHON) -m pytest -q

assets:
	$(PYTHON) -m uav_mec.generate_figures_v2 \
		--canonical $(CANONICAL) \
		--figures figures/thesis \
		--macros paper/generated_results.tex \
		--statistics-table paper/generated_statistics.tex \
		--robustness-table paper/generated_robustness.tex

paper: assets
	mkdir -p build/paper
	$(TECTONIC) -X compile paper/main_v2.tex --outdir build/paper --reruns 2

thesis: assets
	mkdir -p build/thesis
	$(TECTONIC) -X compile thesis/main.tex --outdir build/thesis --reruns 2

verify: test paper thesis
