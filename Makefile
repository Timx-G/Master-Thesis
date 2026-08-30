PYTHON ?= $(if $(wildcard .venv311/bin/python),.venv311/bin/python,python3)
TECTONIC ?= tectonic
CANONICAL ?= artifacts/canonical/v1

.PHONY: test smoke exact-smoke sharded-smoke capacity-dry-run assets paper thesis verify

RUN_TAG ?= $(shell date +%Y%m%d-%H%M%S)

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m uav_mec.experiment \
		--config configs/thesis/smoke.toml \
		--output artifacts/smoke/main-$(RUN_TAG)

exact-smoke:
	$(PYTHON) -m uav_mec.run_exact_benchmark \
		--config configs/thesis/smoke_exact.toml \
		--output artifacts/smoke/exact-$(RUN_TAG)

sharded-smoke:
	$(PYTHON) -m uav_mec.run_shards \
		--config configs/thesis/smoke.toml \
		--output artifacts/smoke/sharded-$(RUN_TAG) \
		--shard-size 1

capacity-dry-run:
	$(PYTHON) -m uav_mec.run_thesis_suite \
		--suite capacity \
		--output artifacts/exploratory/capacity-dry-$(RUN_TAG) \
		--workers 1 \
		--seed-count 2 \
		--development-seed-start 42 \
		--num-slots 3

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
