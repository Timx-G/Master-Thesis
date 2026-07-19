PYTHON ?= python
TECTONIC ?= tectonic
CANONICAL ?= artifacts/canonical/v1

.PHONY: test semantic-gate assets paper thesis verify

test:
	$(PYTHON) -m pytest -q

semantic-gate:
	$(PYTHON) -m pytest tests/test_latency.py tests/test_domain.py tests/test_trace.py tests/test_schedule.py tests/test_schedule_closure.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_evaluation.py tests/test_search.py tests/test_project_metadata.py -q

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
