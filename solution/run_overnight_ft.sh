#!/usr/bin/env bash
# Ночной прогон двух дообучений реранкера (labeled → synthetic), последовательно.
#
# Устойчивость: падение одного эксперимента НЕ прерывает следующий; у каждого
# свой лог и свой artifacts_dir (веса не смешиваются). Перед каждым запуском —
# ожидание свободной VRAM. В конце — сводное сравнение compare_rerankers.py.
#
#   cd solution && nohup bash run_overnight_ft.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="overnight_logs_$STAMP"
mkdir -p "$LOGDIR"
log() { echo "[$(date '+%F %T')] $*" >> "$LOGDIR/runner.log"; }

wait_gpu() {  # ждать ≥9000 МиБ свободной VRAM (до 30 мин), чтобы не стартовать в занятый GPU
    for _ in $(seq 1 60); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "${free:-0}" -ge 9000 ] && return 0
        sleep 30
    done
    log "WARN: GPU так и не освободился (free=${free:-?} MiB) — запускаю как есть"
}

run_exp() {  # $1 = имя эксперимента (без .yaml)
    local name=$1
    wait_gpu
    log "START $name"
    # Без timeout: лучше дать долгому прогону дойти, чем убить почти готовый.
    if python3 -m support_search.cli all --experiment "$name.yaml" > "$LOGDIR/$name.log" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name (rc=$?) — подробности в $LOGDIR/$name.log"
    fi
    # ключевые метрики прогона — сразу в сводный лог
    grep -E "reranker_zs/dev_oof|reranker_ft/dev_oof|выбран|не бьёт" "$LOGDIR/$name.log" >> "$LOGDIR/runner.log" 2>/dev/null
}

log "=== ночной прогон: 2 дообучения bge-reranker ==="
run_exp rerank_bge_ft_labeled
run_exp rerank_bge_ft_synth

python3 compare_rerankers.py > "$LOGDIR/compare.log" 2>&1 \
    && log "compare (zs): ok (rerank_compare/*.csv)" \
    || log "compare (zs): FAIL — см. $LOGDIR/compare.log"
python3 compare_rerankers.py --target-retriever reranker_ft --out rerank_compare_ft > "$LOGDIR/compare_ft.log" 2>&1 \
    && log "compare (ft): ok (rerank_compare_ft/*.csv)" \
    || log "compare (ft): FAIL — см. $LOGDIR/compare_ft.log"
log "=== ГОТОВО. Сводка: этот файл; полные логи: $LOGDIR/*.log ==="
