#!/bin/bash

# Benchmark Script: CPU vs. GPU Preprocessing Performance
# Vergleicht Ausführungszeiten von preprocessing_cpu und preprocessing_cuda

set -e

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

calc_diff() {
    awk -v start="$1" -v end="$2" 'BEGIN { printf "%.3f", end - start }'
}

calc_sum() {
    awk -v left="$1" -v right="$2" 'BEGIN { printf "%.3f", left + right }'
}

calc_ratio() {
    awk -v numerator="$1" -v denominator="$2" 'BEGIN { if (denominator == 0) { printf "inf" } else { printf "%.2f", numerator / denominator } }'
}

# Verzeichnisse
WORK_DIR="$(pwd)"
DATA_DIR="../../data"

# Finde Bilder
IMAGES=($(find "$DATA_DIR" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.png" \) | sort))

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo -e "${RED}Fehler: Keine Bilder in $DATA_DIR gefunden${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     CPU vs. GPU Preprocessing Benchmark            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Überprüfe ob Programme existieren
if [ ! -f "./preprocessing_cpu" ]; then
    echo -e "${RED}Fehler: preprocessing_cpu nicht gefunden${NC}"
    exit 1
fi

if [ ! -f "./preprocessing_cuda" ]; then
    echo -e "${RED}Fehler: preprocessing_cuda nicht gefunden${NC}"
    exit 1
fi

# Teste nur die ersten 3 Bilder (schneller)
IMAGES_TO_TEST=("${IMAGES[@]:0:3}")

# Array für Ergebnisse
declare -a CPU_TIMES
declare -a GPU_TIMES
declare -a IMAGE_NAMES

echo -e "${YELLOW}Teste ${#IMAGES_TO_TEST[@]} Bilder:${NC}"
echo ""

for idx in "${!IMAGES_TO_TEST[@]}"; do
    IMAGE="${IMAGES_TO_TEST[$idx]}"
    BASENAME=$(basename "$IMAGE")
    IMAGE_NAMES[$idx]="$BASENAME"
    
    echo -e "${BLUE}[$(($idx+1))/${#IMAGES_TO_TEST[@]}]${NC} $BASENAME"
    
    # CPU Benchmark
    echo -n "  CPU: "
    START=$(date +%s.%N)
    ./preprocessing_cpu "$IMAGE" > /dev/null 2>&1
    END=$(date +%s.%N)
    CPU_TIME=$(calc_diff "$START" "$END")
    CPU_TIMES[$idx]="$CPU_TIME"
    echo -e "${GREEN}${CPU_TIME}s${NC}"
    
    # GPU Benchmark
    echo -n "  GPU: "
    START=$(date +%s.%N)
    ./preprocessing_cuda "$IMAGE" > /dev/null 2>&1
    END=$(date +%s.%N)
    GPU_TIME=$(calc_diff "$START" "$END")
    GPU_TIMES[$idx]="$GPU_TIME"
    echo -e "${GREEN}${GPU_TIME}s${NC}"
    
    # Speedup berechnen
    SPEEDUP=$(calc_ratio "${CPU_TIMES[$idx]}" "${GPU_TIMES[$idx]}")
    echo -e "  Speedup: ${YELLOW}${SPEEDUP}x${NC}"
    echo ""
done

# Zusammenfassung
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              ZUSAMMENFASSUNG                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Durchschnitte berechnen
CPU_SUM=0
GPU_SUM=0
for i in "${!CPU_TIMES[@]}"; do
    CPU_SUM=$(calc_sum "$CPU_SUM" "${CPU_TIMES[$i]}")
    GPU_SUM=$(calc_sum "$GPU_SUM" "${GPU_TIMES[$i]}")
done

NUM_TESTS=${#CPU_TIMES[@]}
CPU_AVG=$(calc_ratio "$CPU_SUM" "$NUM_TESTS")
GPU_AVG=$(calc_ratio "$GPU_SUM" "$NUM_TESTS")
AVG_SPEEDUP=$(calc_ratio "$CPU_AVG" "$GPU_AVG")

echo -e "Anzahl Tests: ${YELLOW}${NUM_TESTS}${NC}"
echo -e "Durchschnittliche CPU-Zeit: ${GREEN}${CPU_AVG}s${NC}"
echo -e "Durchschnittliche GPU-Zeit: ${GREEN}${GPU_AVG}s${NC}"
echo -e "Durchschnittlicher Speedup: ${YELLOW}${AVG_SPEEDUP}x${NC}"
echo ""

# Tabelle
echo -e "${BLUE}Detaillierte Ergebnisse:${NC}"
printf "%-30s %12s %12s %12s\n" "Bild" "CPU (s)" "GPU (s)" "Speedup"
printf "%-30s %12s %12s %12s\n" "$(printf '─%.0s' {1..30})" "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..12})"

for i in "${!CPU_TIMES[@]}"; do
    SPEEDUP=$(calc_ratio "${CPU_TIMES[$i]}" "${GPU_TIMES[$i]}")
    printf "%-30s %12s %12s %12sx\n" "${IMAGE_NAMES[$i]}" "${CPU_TIMES[$i]}" "${GPU_TIMES[$i]}" "$SPEEDUP"
done

echo ""
echo -e "${GREEN}✓ Benchmark abgeschlossen${NC}"
