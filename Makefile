# Targets that need no network are marked [offline].
.PHONY: all test audit a3 partb vocab corpus a3-full clean

all: test audit a3 partb          ## [offline] everything reproducible here

vocab:                            ## re-fetch + verify the GPT-2 vocab
	./tools/fetch_gpt2_vocab.sh

test:                             ## [offline] 19 regression tests
	python3 -m pytest partA/tests/test_audit.py -q

audit:                            ## [offline] A2 ablations -> partA/results/a2_audit.txt
	cd partA && python3 audit_experiments.py

a3:                               ## [offline] A3 on the SMOKE corpus (not eval numbers)
	cd partA && python3 train_spm.py --vocab 1000 --out vendor/spm_multi
	cd partA && python3 run_a3.py --tokenizers gpt2,bytes,spm:vendor/spm_multi.model

partb:                            ## [offline] B1-B4 -> partB/results/partB_output.txt
	cd partB && python3 capacity.py

corpus:                           ## NEEDS NETWORK: download FLORES-200 dev
	cd partA && python3 build_corpus.py --out corpora/flores --split dev

a3-full: corpus                   ## NEEDS NETWORK: the real A3 table
	cd partA && python3 run_a3.py --corpus corpora/flores \
	  --langs eng,hin,kan,tam,tel,ben,mar \
	  --tokenizers gpt2,bytes,hf:xlm-roberta-base,hf:ai4bharat/IndicBERTv2-MLM-only \
	  --out results/a3_table_flores.txt

repro-v0:                         ## [offline] run the intern's ORIGINAL script unchanged
	cd partA && TIKTOKEN_CACHE_DIR=.tiktoken_cache python3 fertility_v0_reference.py \
	  --corpus eng=corpora/sample/eng_sample.txt \
	  --corpus hin=corpora/sample/hin_sample.txt --tokenizer gpt2

clean:
	rm -rf partA/results partB/results partA/.tiktoken_cache **/__pycache__
