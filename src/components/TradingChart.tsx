// @ts-nocheck
import { useEffect, useRef } from "react";
import { createChart, ColorType, CrosshairMode } from "lightweight-charts";

// Generate realistic mock OHLCV data
function generateMockCandles(basePrice: number, count = 120) {
  const candles = [];
  let price = basePrice * 0.985;
  const now = Math.floor(Date.now() / 1000);
  const interval = 4 * 60 * 60; // 4H candles

  for (let i = count; i >= 0; i--) {
    const volatility = basePrice * 0.003;
    const open = price;
    const change = (Math.random() - 0.48) * volatility;
    const close = open + change;
    const high = Math.max(open, close) + Math.random() * volatility * 0.6;
    const low = Math.min(open, close) - Math.random() * volatility * 0.6;
    candles.push({
      time: now - i * interval,
      open: parseFloat(open.toFixed(5)),
      high: parseFloat(high.toFixed(5)),
      low: parseFloat(low.toFixed(5)),
      close: parseFloat(close.toFixed(5)),
    });
    price = close;
  }
  return candles;
}

// Draw probability cone as overlay on the chart pane
function drawCone(canvas, chart, series, prediction, horizon) {
  if (!canvas || !chart || !series || !prediction) return;

  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const horizonData = prediction.horizons[horizon];
  if (!horizonData) return;

  const isUp = horizonData.direction === "bullish";
  const isNeutral = horizonData.direction === "neutral";
  const color = isNeutral ? "rgba(160,168,184," : isUp ? "rgba(79,195,247," : "rgba(255,71,87,";

  // Get last bar time and price
  const data = series.data();
  if (!data || data.length === 0) return;
  const lastBar = data[data.length - 1];

  // Convert last bar time to x coordinate
  const lastX = chart.timeScale().timeToCoordinate(lastBar.time);
  const lastY = series.priceToCoordinate(lastBar.close);
  if (lastX === null || lastY === null) return;

  // Calculate future time point based on horizon
  const horizonSeconds = {
    "1H": 3600,
    "4H": 14400,
    "1D": 86400,
    "7D": 604800,
  };
  const futureTime = lastBar.time + (horizonSeconds[horizon] || 86400);
  const futureX = chart.timeScale().timeToCoordinate(futureTime);

  // If future time is off screen, draw to edge
  const endX = futureX !== null ? futureX : W;
  if (endX <= lastX) return;

  const moveAmount = Math.abs(horizonData.expectedMove) / 100 * lastBar.close;
  const midPrice = lastBar.close + (isUp ? moveAmount : -moveAmount) * (isNeutral ? 0 : 1);
  const upperPrice = lastBar.close + moveAmount * 1.7;
  const lowerPrice = lastBar.close + (isUp ? moveAmount * 0.2 : -moveAmount * 1.7);

  const midY = series.priceToCoordinate(midPrice);
  const upperY = series.priceToCoordinate(upperPrice);
  const lowerY = series.priceToCoordinate(lowerPrice);

  if (midY === null || upperY === null || lowerY === null) return;

  // Cone fill
  const grad = ctx.createLinearGradient(lastX, 0, endX, 0);
  grad.addColorStop(0, `${color}0)`);
  grad.addColorStop(1, `${color}0.12)`);

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(endX, upperY);
  ctx.lineTo(endX, lowerY);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Upper bound line
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(endX, upperY);
  ctx.strokeStyle = `${color}0.2)`;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  ctx.stroke();

  // Lower bound line
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(endX, lowerY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Median dashed line
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(endX, midY);
  ctx.strokeStyle = `${color}0.8)`;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Probability label
  const prob = Math.round(horizonData.probability * 100);
  const label = `${isNeutral ? "—" : isUp ? "▲" : "▼"} ${prob}%`;
  ctx.font = "bold 11px 'DM Mono', monospace";
  const textW = ctx.measureText(label).width;

  ctx.fillStyle = `${color}0.15)`;
  ctx.beginPath();
  ctx.roundRect(endX - textW - 16, midY - 10, textW + 12, 20, 4);
  ctx.fill();

  ctx.fillStyle = `${color}1)`;
  ctx.fillText(label, endX - textW - 10, midY + 4);
}

export default function TradingChart({ pair, prediction, horizon }) {
  const chartContainerRef = useRef(null);
  const canvasOverlayRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const resizeObserverRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;

    // Create chart
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: "#080c14" },
        textColor: "rgba(255,255,255,0.25)",
        fontFamily: "'DM Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: "rgba(255,255,255,0.15)",
          labelBackgroundColor: "#1a2332",
        },
        horzLine: {
          color: "rgba(255,255,255,0.15)",
          labelBackgroundColor: "#1a2332",
        },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.06)",
        textColor: "rgba(255,255,255,0.25)",
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.06)",
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time) => {
          const date = new Date(time * 1000);
          return `${date.getMonth() + 1}/${date.getDate()}`;
        },
      },
      handleScroll: true,
      handleScale: true,
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#4fc3f7",
      downColor: "#ff4757",
      borderUpColor: "#4fc3f7",
      borderDownColor: "#ff4757",
      wickUpColor: "rgba(79,195,247,0.6)",
      wickDownColor: "rgba(255,71,87,0.6)",
    });

    // Load mock data
    const basePrice = prediction?.price || 1.08432;
    const mockData = generateMockCandles(basePrice);
    candleSeries.setData(mockData);

    chart.timeScale().fitContent();

    chartRef.current = chart;
    seriesRef.current = candleSeries;

    // Resize observer
    resizeObserverRef.current = new ResizeObserver(() => {
      chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight,
      });
      // Redraw cone after resize
      setTimeout(() => redrawCone(), 50);
    });
    resizeObserverRef.current.observe(container);

    // Initial size
    chart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });

    return () => {
      resizeObserverRef.current?.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [pair]);

  // Redraw cone when prediction or horizon changes
  const redrawCone = () => {
    if (!canvasOverlayRef.current || !chartRef.current || !seriesRef.current) return;
    drawCone(canvasOverlayRef.current, chartRef.current, seriesRef.current, prediction, horizon);
  };

  useEffect(() => {
    // Small delay to ensure chart has rendered
    const timeout = setTimeout(redrawCone, 100);
    return () => clearTimeout(timeout);
  }, [prediction, horizon, pair]);

  // Sync overlay canvas size with chart container
  useEffect(() => {
    const container = chartContainerRef.current;
    const overlay = canvasOverlayRef.current;
    if (!container || !overlay) return;

    const sync = () => {
      overlay.width = container.clientWidth;
      overlay.height = container.clientHeight;
      redrawCone();
    };

    const observer = new ResizeObserver(sync);
    observer.observe(container);
    sync();
    return () => observer.disconnect();
  }, [prediction, horizon]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {/* Lightweight Charts renders here */}
      <div ref={chartContainerRef} style={{ width: "100%", height: "100%" }} />

      {/* Cone overlay canvas */}
      <canvas
        ref={canvasOverlayRef}
        style={{
          position: "absolute",
          top: 0, left: 0,
          pointerEvents: "none",
          width: "100%",
          height: "100%",
        }}
      />

      {/* Chart label */}
      <div style={{
        position: "absolute", top: "12px", left: "14px",
        fontSize: "10px", fontFamily: "'DM Mono', monospace",
        color: "rgba(255,255,255,0.2)", letterSpacing: "0.08em",
        pointerEvents: "none"
      }}>
        {pair} · 4H · Projection: {horizon}
      </div>
    </div>
  );
}