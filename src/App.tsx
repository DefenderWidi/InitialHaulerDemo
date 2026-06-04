import { useEffect, useMemo, useRef, useState } from "react";

type Status = "UNDERLOAD" | "IDEAL" | "OVERLOAD"
type ValidationStatus = "PENDING" | "VALID" | "INVALID"
type Severity = "LOW" | "MEDIUM" | "HIGH"
type PageMode = "LIVE" | "VALIDATION"
type EvidenceKind = "snapshot" | "crop"

type DetectionPrediction = {
  x: number
  y: number
  width: number
  height: number
  confidence: number
  class: string
}

type ClassificationPrediction = {
  class: string
  confidence: number
}

type DetectionResponse = {
  predictions?: DetectionPrediction[]
  image?: {
    width: number
    height: number
  }
}

type RawClassificationResponse = {
  predictions?: ClassificationPrediction[] | Record<string, number | { confidence?: number }>
  top?: string
  confidence?: number
  predicted_classes?: string[]
}

type CapturedFrame = {
  dataUrl: string
  videoTime: number
  sourceWidth: number
  sourceHeight: number
  width: number
  height: number
  scaleX: number
  scaleY: number
}

type DetectionResult = {
  status: Status
  confidence: number
  detectorConfidence: number
  detectorLatencyMs: number
  classifierLatencyMs: number
  notes: string
  recommendation: string
  severity: Severity
  validation: ValidationStatus
  validatedAs?: Status
  validatorNote?: string
  timestamp: string
  videoTime: number
  snapshotUrl: string
  cropUrl: string
  detectionBox: DetectionPrediction
}

type HistoryItem = DetectionResult & {
  id: number
  cctv: string
  source: string
  totalDetection: number
}

type EvidencePreview = {
  title: string
  subtitle: string
  imageUrl: string
  kind: EvidenceKind
  item: HistoryItem
}

const VIDEO_SOURCE = "/ME112.mp4"

/**
 * OPTIMASI LATENCY + EVIDENCE EVENT:
 * - Detector memakai frame kecil agar inference lebih cepat.
 * - Evidence tetap memakai frame lebih besar agar bukti validasi jelas.
 * - Evidence tidak disimpan setiap frame.
 * - Evidence disimpan saat cooldown sudah lewat, confidence cukup,
 *   dan hauler dianggap event baru berdasarkan:
 *   1. hauler baru muncul setelah sebelumnya hilang, atau
 *   2. posisi hauler cukup berbeda dari evidence terakhir.
 */
const DETECT_MAX_WIDTH = 512
const DETECT_JPEG_QUALITY = 0.55

const EVIDENCE_MAX_WIDTH = 960
const EVIDENCE_JPEG_QUALITY = 0.78
const CROP_JPEG_QUALITY = 0.72

/**
 * Interval scan detector.
 * Makin kecil = lebih responsif, tapi request Roboflow lebih sering.
 */
const DETECTION_SCAN_IDLE_MS = 100

/**
 * Confidence minimal agar box hauler ditampilkan di layar.
 * Jangan terlalu tinggi karena dataset detector masih kecil.
 */
const MIN_VISUAL_CONFIDENCE = 0.20

/**
 * Confidence minimal agar hauler boleh masuk evidence.
 * Untuk demo + human validation, 0.25 lebih aman daripada terlalu ketat.
 */
const MIN_CAPTURE_CONFIDENCE = 0.25

/**
 * Jarak minimal antar evidence.
 * 3000 ms = maksimal sekitar 1 evidence tiap 3 detik.
 * Kalau masih terlalu jarang, turunkan ke 2500.
 * Kalau terlalu banyak, naikkan ke 4000.
 */
const CAPTURE_COOLDOWN_MS = 3000

/**
 * Kalau hauler tidak terlihat selama waktu ini,
 * sistem menganggap event sebelumnya sudah selesai.
 */
const LOST_RESET_MS = 1200

/**
 * Box visual tetap ditahan sebentar supaya tidak kedip-kedip.
 */
const VISUAL_HOLD_MS = 2200

/**
 * Timeout request ke Roboflow.
 */
const REQUEST_TIMEOUT_MS = 5000

/**
 * Batas maksimal pending evidence agar UI tidak terlalu berat.
 */
const MAX_PENDING_ITEMS = 40

/**
 * Threshold untuk membedakan apakah box sekarang adalah event hauler baru.
 *
 * IoU kecil = posisi box jauh berbeda dari evidence terakhir.
 * Center shift besar = titik tengah box berpindah jauh.
 */
const NEW_EVENT_IOU_THRESHOLD = 0.25
const NEW_EVENT_CENTER_SHIFT_RATIO = 0.60

const cctvList = ["CCTV - DisposalMandalikaHR", "CCTV - PTZ", "CCTV - Front Road Disposal"]
const statusOptions: Status[] = ["UNDERLOAD", "IDEAL", "OVERLOAD"]

const ROBOFLOW_API_KEY = import.meta.env.VITE_ROBOFLOW_API_KEY as string | undefined

const DETECT_PROJECT = (import.meta.env.VITE_DETECT_PROJECT as string | undefined) || "hauler-detector-cctv-new-pov"
const DETECT_VERSION = (import.meta.env.VITE_DETECT_VERSION as string | undefined) || "2"

const CLASSIFY_PROJECT = (import.meta.env.VITE_CLASSIFY_PROJECT as string | undefined) || "hauler-load-classifier-cctv-new"
const CLASSIFY_VERSION = (import.meta.env.VITE_CLASSIFY_VERSION as string | undefined) || "1"

function normalizeStatus(label: string): Status {
  const value = label.trim().toLowerCase()
  if (value.includes("under")) return "UNDERLOAD"
  if (value.includes("over")) return "OVERLOAD"
  return "IDEAL"
}

function formatStatus(status: Status) {
  if (status === "UNDERLOAD") return "Under Load"
  if (status === "OVERLOAD") return "Over Load"
  return "Ideal"
}

function formatVideoTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00"
  const minute = Math.floor(seconds / 60)
  const second = Math.floor(seconds % 60)
  return `${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}`
}

function getStatusStyle(status: Status) {
  if (status === "UNDERLOAD") {
    return {
      chip: "border-red-200 bg-red-50 text-red-700",
      solid: "bg-red-600 text-white",
      title: "text-red-600",
      border: "border-red-500",
      soft: "bg-red-50",
      gradient: "from-red-500 to-rose-600",
      ring: "ring-red-200",
    }
  }

  if (status === "IDEAL") {
    return {
      chip: "border-emerald-200 bg-emerald-50 text-emerald-700",
      solid: "bg-emerald-600 text-white",
      title: "text-emerald-700",
      border: "border-emerald-500",
      soft: "bg-emerald-50",
      gradient: "from-emerald-500 to-green-700",
      ring: "ring-emerald-200",
    }
  }

  return {
    chip: "border-amber-200 bg-amber-50 text-amber-700",
    solid: "bg-amber-500 text-white",
    title: "text-amber-700",
    border: "border-amber-500",
    soft: "bg-amber-50",
    gradient: "from-amber-400 to-orange-600",
    ring: "ring-amber-200",
  }
}

function getSeverity(status: Status, confidence: number): Severity {
  if (status === "IDEAL" && confidence >= 70) return "LOW"
  if (confidence < 55) return "MEDIUM"
  if (status === "OVERLOAD" || status === "UNDERLOAD") return "HIGH"
  return "MEDIUM"
}

function getRecommendation(status: Status, confidence: number) {
  if (confidence < 55) {
    return "Confidence model rendah. Evidence perlu dicek manual oleh pengawas sebelum dipakai sebagai dasar keputusan."
  }

  if (status === "OVERLOAD") {
    return "Indikasi over load. Prioritaskan validasi PJA dan pastikan muatan tidak berpotensi tumpah saat hauling."
  }

  if (status === "UNDERLOAD") {
    return "Indikasi under load. Perlu validasi terhadap loading practice dan potensi kehilangan produktivitas."
  }

  return "Muatan terbaca ideal. Simpan sebagai evidence valid jika pengawas menyetujui hasil model."
}

function getTimestamp() {
  return new Intl.DateTimeFormat("id-ID", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date())
}

function toPercent(confidence: number) {
  return confidence > 1 ? confidence : confidence * 100
}

function stripBase64Prefix(dataUrl: string) {
  return dataUrl.replace(/^data:image\/[a-zA-Z]+;base64,/, "")
}

function ensureRoboflowEnv() {
  if (!ROBOFLOW_API_KEY) {
    throw new Error("VITE_ROBOFLOW_API_KEY belum diisi di file .env")
  }

  if (!DETECT_PROJECT || !DETECT_VERSION || !CLASSIFY_PROJECT || !CLASSIFY_VERSION) {
    throw new Error("Env Roboflow belum lengkap. Cek VITE_DETECT_PROJECT, VITE_DETECT_VERSION, VITE_CLASSIFY_PROJECT, VITE_CLASSIFY_VERSION.")
  }
}

async function inferRoboflow(project: string, version: string, imageDataUrl: string, confidence = 0.4) {
  ensureRoboflowEnv()

  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const url =
      `https://serverless.roboflow.com/${project}/${version}` +
      `?api_key=${ROBOFLOW_API_KEY}` +
      `&confidence=${confidence}` +
      `&format=json` +
      `&image_type=base64`

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: stripBase64Prefix(imageDataUrl),
      signal: controller.signal,
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(`Roboflow inference gagal: ${response.status} ${text}`)
    }

    return response.json()
  } finally {
    window.clearTimeout(timeoutId)
  }
}

function normalizeClassificationResponse(result: RawClassificationResponse): ClassificationPrediction | null {
  if (Array.isArray(result.predictions)) {
    return [...result.predictions].sort((a, b) => b.confidence - a.confidence)[0] ?? null
  }

  if (result.predictions && typeof result.predictions === "object") {
    const mapped = Object.entries(result.predictions).map(([label, value]) => {
      if (typeof value === "number") {
        return { class: label, confidence: value }
      }

      return { class: label, confidence: value.confidence ?? 0 }
    })

    return mapped.sort((a, b) => b.confidence - a.confidence)[0] ?? null
  }

  if (result.top) {
    return { class: result.top, confidence: result.confidence ?? 0 }
  }

  if (result.predicted_classes?.length) {
    return { class: result.predicted_classes[0], confidence: result.confidence ?? 0 }
  }

  return null
}

async function detectHauler(frameDataUrl: string) {
  const startedAt = performance.now()

  const result = (await inferRoboflow(DETECT_PROJECT, DETECT_VERSION, frameDataUrl, MIN_VISUAL_CONFIDENCE)) as DetectionResponse
  const latencyMs = Math.round(performance.now() - startedAt)

  const all = result.predictions ?? []

  const haulers = all
    .filter((prediction) => prediction.class.toLowerCase().includes("hauler"))
    .sort((a, b) => b.confidence - a.confidence)

  return {
    best: haulers[0] ?? null,
    all,
    latencyMs,
  }
}

async function classifyLoad(cropDataUrl: string) {
  const startedAt = performance.now()

  const result = (await inferRoboflow(CLASSIFY_PROJECT, CLASSIFY_VERSION, cropDataUrl, 0.0)) as RawClassificationResponse
  const latencyMs = Math.round(performance.now() - startedAt)

  return {
    prediction: normalizeClassificationResponse(result),
    latencyMs,
  }
}

function scaleBox(box: DetectionPrediction, scaleX: number, scaleY: number): DetectionPrediction {
  return {
    ...box,
    x: box.x * scaleX,
    y: box.y * scaleY,
    width: box.width * scaleX,
    height: box.height * scaleY,
  }
}

function getBoxRect(box: DetectionPrediction) {
  const left = box.x - box.width / 2
  const top = box.y - box.height / 2
  const right = box.x + box.width / 2
  const bottom = box.y + box.height / 2

  return { left, top, right, bottom }
}

function calculateIoU(a: DetectionPrediction, b: DetectionPrediction) {
  const boxA = getBoxRect(a)
  const boxB = getBoxRect(b)

  const interLeft = Math.max(boxA.left, boxB.left)
  const interTop = Math.max(boxA.top, boxB.top)
  const interRight = Math.min(boxA.right, boxB.right)
  const interBottom = Math.min(boxA.bottom, boxB.bottom)

  const interWidth = Math.max(0, interRight - interLeft)
  const interHeight = Math.max(0, interBottom - interTop)
  const intersection = interWidth * interHeight

  const areaA = Math.max(1, a.width * a.height)
  const areaB = Math.max(1, b.width * b.height)
  const union = areaA + areaB - intersection

  return intersection / Math.max(1, union)
}

function isDifferentHaulerEvent(current: DetectionPrediction, previous: DetectionPrediction | null) {
  if (!previous) return true

  const iou = calculateIoU(current, previous)

  const dx = current.x - previous.x
  const dy = current.y - previous.y
  const centerDistance = Math.sqrt(dx * dx + dy * dy)

  const referenceSize = Math.max(previous.width, previous.height, current.width, current.height, 1)
  const centerShiftRatio = centerDistance / referenceSize

  return iou < NEW_EVENT_IOU_THRESHOLD || centerShiftRatio > NEW_EVENT_CENTER_SHIFT_RATIO
}

function captureVideoFrame(video: HTMLVideoElement, maxWidth: number, quality: number): CapturedFrame {
  const sourceWidth = video.videoWidth
  const sourceHeight = video.videoHeight

  if (!sourceWidth || !sourceHeight) {
    throw new Error("Video belum siap. Tunggu sampai video tampil, lalu jalankan deteksi lagi.")
  }

  const scale = Math.min(1, maxWidth / sourceWidth)
  const width = Math.max(1, Math.round(sourceWidth * scale))
  const height = Math.max(1, Math.round(sourceHeight * scale))

  const canvas = document.createElement("canvas")
  const ctx = canvas.getContext("2d")

  if (!ctx) {
    throw new Error("Canvas context tidak tersedia di browser.")
  }

  canvas.width = width
  canvas.height = height
  ctx.drawImage(video, 0, 0, width, height)

  return {
    dataUrl: canvas.toDataURL("image/jpeg", quality),
    videoTime: video.currentTime,
    sourceWidth,
    sourceHeight,
    width,
    height,
    scaleX: width / sourceWidth,
    scaleY: height / sourceHeight,
  }
}

function cropByDetection(sourceDataUrl: string, box: DetectionPrediction, paddingRatio = 0.08): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()

    img.onload = () => {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")

      if (!ctx) {
        reject(new Error("Canvas context tidak tersedia saat crop."))
        return
      }

      const padX = box.width * paddingRatio
      const padY = box.height * paddingRatio

      const left = Math.max(0, box.x - box.width / 2 - padX)
      const top = Math.max(0, box.y - box.height / 2 - padY)
      const right = Math.min(img.width, box.x + box.width / 2 + padX)
      const bottom = Math.min(img.height, box.y + box.height / 2 + padY)

      const cropWidth = Math.max(1, right - left)
      const cropHeight = Math.max(1, bottom - top)

      canvas.width = cropWidth
      canvas.height = cropHeight
      ctx.drawImage(img, left, top, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight)

      resolve(canvas.toDataURL("image/jpeg", CROP_JPEG_QUALITY))
    }

    img.onerror = () => reject(new Error("Gagal membaca image untuk proses crop."))
    img.src = sourceDataUrl
  })
}

function StatCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/75 p-4 shadow-[0_16px_45px_rgba(15,23,42,0.06)] backdrop-blur-xl transition hover:-translate-y-0.5 hover:shadow-[0_20px_55px_rgba(22,163,74,0.10)]">
      <div className="text-[11px] font-black uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-black text-slate-900">{value}</div>
      <div className="mt-1 text-xs font-semibold text-slate-500">{sub}</div>
    </div>
  )
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/65 px-4 py-3 shadow-inner backdrop-blur-xl">
      <div className="text-[11px] font-black uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-sm font-black text-slate-800">{value}</div>
    </div>
  )
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)

const scanTimerRef = useRef<number | null>(null)
const isScanningRef = useRef(false)
const isProcessingRef = useRef(false)
const autoStartedRef = useRef(false)
const haulerInFrameRef = useRef(false)
const lastCaptureAtRef = useRef(0)
const lastSeenAtRef = useRef(0)
const lastCapturedBoxRef = useRef<DetectionPrediction | null>(null)
const clearVisualTimerRef = useRef<number | null>(null)

  const [pageMode, setPageMode] = useState<PageMode>("LIVE")
  const [selectedCctv, setSelectedCctv] = useState(cctvList[0])
  const [isScanning, setIsScanning] = useState(false)
  const [isProcessingDetection, setIsProcessingDetection] = useState(false)
  const [isClassifying, setIsClassifying] = useState(false)

  const [latestHaulerBox, setLatestHaulerBox] = useState<DetectionPrediction | null>(null)
  const [latestBoxAt, setLatestBoxAt] = useState(0)
  const [frameInfo, setFrameInfo] = useState({ width: 1280, height: 720 })

  const [lastDetectorLatency, setLastDetectorLatency] = useState<number | null>(null)
  const [lastClassifierLatency, setLastClassifierLatency] = useState<number | null>(null)
  const [lastRoundTripLatency, setLastRoundTripLatency] = useState<number | null>(null)

  const [scanMessage, setScanMessage] = useState("Standby")
  const [error, setError] = useState("")
  const [latestResult, setLatestResult] = useState<HistoryItem | null>(null)
  const [lastSnapshot, setLastSnapshot] = useState("")
  const [lastCrop, setLastCrop] = useState("")

  const [pendingDetections, setPendingDetections] = useState<HistoryItem[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [filterStatus, setFilterStatus] = useState<"ALL" | Status>("ALL")
  const [correctionStatus, setCorrectionStatus] = useState<Record<number, Status>>({})
  const [validatorNotes, setValidatorNotes] = useState<Record<number, string>>({})
  const [preview, setPreview] = useState<EvidencePreview | null>(null)

  const summary = useMemo(() => {
    const total = history.length
    const valid = history.filter((item) => item.validation === "VALID").length
    const invalid = history.filter((item) => item.validation === "INVALID").length
    const agreement = total ? Math.round((valid / total) * 100) : 0

    return {
      pending: pendingDetections.length,
      total,
      valid,
      invalid,
      agreement,
    }
  }, [history, pendingDetections])

  const filteredHistory = useMemo(() => {
    if (filterStatus === "ALL") return history

    return history.filter((item) => (item.validatedAs || item.status) === filterStatus)
  }, [filterStatus, history])

  const overlayBox =
    latestHaulerBox && Date.now() - latestBoxAt <= VISUAL_HOLD_MS
      ? {
          left: ((latestHaulerBox.x - latestHaulerBox.width / 2) / frameInfo.width) * 100,
          top: ((latestHaulerBox.y - latestHaulerBox.height / 2) / frameInfo.height) * 100,
          width: (latestHaulerBox.width / frameInfo.width) * 100,
          height: (latestHaulerBox.height / frameInfo.height) * 100,
        }
      : null

  const scheduleNextScan = (delay = DETECTION_SCAN_IDLE_MS) => {
    if (!isScanningRef.current) return

    if (scanTimerRef.current) {
      window.clearTimeout(scanTimerRef.current)
    }

    scanTimerRef.current = window.setTimeout(() => {
      scanOnce()
    }, delay)
  }

  const scheduleVisualClear = () => {
    if (clearVisualTimerRef.current) {
      window.clearTimeout(clearVisualTimerRef.current)
    }

    clearVisualTimerRef.current = window.setTimeout(() => {
      setLatestHaulerBox(null)
    }, VISUAL_HOLD_MS + 100)
  }

  const queueDetectedHauler = async (
    detectionBoxOriginal: DetectionPrediction,
    totalDetection: number,
    detectorLatencyMs: number
  ) => {
    const video = videoRef.current
    if (!video) return

    const roundStartedAt = performance.now()
    setIsClassifying(true)

    try {
      const evidenceFrame = captureVideoFrame(video, EVIDENCE_MAX_WIDTH, EVIDENCE_JPEG_QUALITY)
      const boxOnEvidence = scaleBox(detectionBoxOriginal, evidenceFrame.scaleX, evidenceFrame.scaleY)
      const cropUrl = await cropByDetection(evidenceFrame.dataUrl, boxOnEvidence, 0.08)

      const classification = await classifyLoad(cropUrl)
      const prediction = classification.prediction

      if (!prediction) {
        throw new Error("Hauler terdeteksi, tetapi classifier tidak mengembalikan label underload/ideal/overload.")
      }

      const status = normalizeStatus(prediction.class)
      const confidence = toPercent(prediction.confidence)
      const detectorConfidence = toPercent(detectionBoxOriginal.confidence)
      const classifierLatencyMs = classification.latencyMs
      const roundTripLatencyMs = Math.round(performance.now() - roundStartedAt)

      setLastClassifierLatency(classifierLatencyMs)
      setLastRoundTripLatency(roundTripLatencyMs)

      const item: HistoryItem = {
        id: Date.now(),
        cctv: selectedCctv,
        source: VIDEO_SOURCE,
        totalDetection,
        status,
        confidence,
        detectorConfidence,
        detectorLatencyMs,
        classifierLatencyMs,
        severity: getSeverity(status, confidence),
        validation: "PENDING",
        timestamp: getTimestamp(),
        videoTime: evidenceFrame.videoTime,
        snapshotUrl: evidenceFrame.dataUrl,
        cropUrl,
        detectionBox: detectionBoxOriginal,
        notes: `Hauler terdeteksi pada detik ${formatVideoTime(evidenceFrame.videoTime)}. Sistem langsung menyimpan screenshot, melakukan crop area hauler, lalu classifier membaca muatan sebagai ${formatStatus(
          status
        )} dengan confidence ${confidence.toFixed(1)}%.`,
        recommendation: getRecommendation(status, confidence),
      }

      setLatestResult(item)
      setLastSnapshot(evidenceFrame.dataUrl)
      setLastCrop(cropUrl)
      setCorrectionStatus((prev) => ({ ...prev, [item.id]: status }))
      setPendingDetections((prev) => [item, ...prev].slice(0, MAX_PENDING_ITEMS))
      setScanMessage(`Evidence masuk validasi: ${formatStatus(status)} (${confidence.toFixed(1)}%)`)
    } finally {
      setIsClassifying(false)
    }
  }

  const scanOnce = async () => {
    if (!isScanningRef.current) return

    if (isProcessingRef.current) {
      scheduleNextScan(60)
      return
    }

    const video = videoRef.current
    if (!video) {
      scheduleNextScan()
      return
    }

    try {
      isProcessingRef.current = true
      setIsProcessingDetection(true)
      setError("")

      const detectorFrame = captureVideoFrame(video, DETECT_MAX_WIDTH, DETECT_JPEG_QUALITY)
      const detection = await detectHauler(detectorFrame.dataUrl)

      setLastDetectorLatency(detection.latencyMs)

if (!detection.best) {
  const now = Date.now()
  setScanMessage(`Scanning... belum ada hauler • detector ${detection.latencyMs}ms`)

  if (now - lastSeenAtRef.current >= LOST_RESET_MS) {
    haulerInFrameRef.current = false
    lastCapturedBoxRef.current = null
  }

  return
}
      const bestOriginal = scaleBox(detection.best, 1 / detectorFrame.scaleX, 1 / detectorFrame.scaleY)
      const now = Date.now()

      lastSeenAtRef.current = now
      setLatestHaulerBox(bestOriginal)
      setLatestBoxAt(now)
      scheduleVisualClear()
      setScanMessage(`Hauler locked • detector ${detection.latencyMs}ms • conf ${toPercent(bestOriginal.confidence).toFixed(1)}%`)

const cooldownPassed = now - lastCaptureAtRef.current >= CAPTURE_COOLDOWN_MS
const isNewDetectionEvent = !haulerInFrameRef.current
const isDifferentFromLastCaptured = isDifferentHaulerEvent(bestOriginal, lastCapturedBoxRef.current)
const isConfidentEnough = bestOriginal.confidence >= MIN_CAPTURE_CONFIDENCE

const shouldCaptureEvidence =
  cooldownPassed &&
  isConfidentEnough &&
  (isNewDetectionEvent || isDifferentFromLastCaptured)

if (shouldCaptureEvidence) {
  haulerInFrameRef.current = true
  lastCaptureAtRef.current = now
  lastCapturedBoxRef.current = bestOriginal

  await queueDetectedHauler(bestOriginal, detection.all.length, detection.latencyMs)
} else {
  haulerInFrameRef.current = true
}
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request Roboflow timeout. Frame terlalu berat atau koneksi lambat. Kode sudah memakai frame 512px; coba ulang beberapa detik lagi.")
      } else {
        setError(err instanceof Error ? err.message : "Terjadi error saat menjalankan deteksi.")
      }
    } finally {
      isProcessingRef.current = false
      setIsProcessingDetection(false)
      scheduleNextScan()
    }
  }

  const startAutoDetection = async () => {
    const video = videoRef.current
    if (!video) return

    try {
      ensureRoboflowEnv()
      setError("")
      setScanMessage("Menjalankan video dan memulai scanning hauler...")
      await video.play()

      isScanningRef.current = true
      setIsScanning(true)
      haulerInFrameRef.current = false
      lastSeenAtRef.current = Date.now()

      if (scanTimerRef.current) {
        window.clearTimeout(scanTimerRef.current)
        scanTimerRef.current = null
      }

      scanOnce()
    } catch (err) {
      isScanningRef.current = false
      setIsScanning(false)
      setError(err instanceof Error ? err.message : "Video tidak dapat dijalankan. Klik Play lalu Start Detection.")
    }
  }

  const stopAutoDetection = () => {
    isScanningRef.current = false
    setIsScanning(false)
    setScanMessage("Standby")
    autoStartedRef.current = false

    if (scanTimerRef.current) {
      window.clearTimeout(scanTimerRef.current)
      scanTimerRef.current = null
    }
  }

  const validateDetection = (id: number, validation: ValidationStatus) => {
    const item = pendingDetections.find((row) => row.id === id)
    if (!item) return

    const validatedAs = validation === "VALID" ? item.status : correctionStatus[id] || item.status
    const validatorNote = validatorNotes[id] || ""
    const updated: HistoryItem = { ...item, validation, validatedAs, validatorNote }

    setHistory((prev) => [updated, ...prev])
    setPendingDetections((prev) => prev.filter((row) => row.id !== id))
  }

  useEffect(() => {
    return () => {
      isScanningRef.current = false

      if (scanTimerRef.current) {
        window.clearTimeout(scanTimerRef.current)
      }

      if (clearVisualTimerRef.current) {
        window.clearTimeout(clearVisualTimerRef.current)
      }
    }
  }, [])

  const latestStyle = latestResult ? getStatusStyle(latestResult.status) : null

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(34,197,94,0.13),transparent_34%),linear-gradient(180deg,#f8fafc_0%,#eef2f7_100%)] text-slate-800">
   <header className="sticky top-0 z-40 border-b border-green-400/20 bg-white/55 shadow-[0_12px_40px_rgba(22,163,74,0.10)] backdrop-blur-2xl">
  <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-green-400/80 to-transparent" />
  <div className="pointer-events-none absolute left-0 top-0 h-full w-24 bg-gradient-to-r from-green-400/20 to-transparent" />
  <div className="pointer-events-none absolute right-0 top-0 h-full w-24 bg-gradient-to-l from-lime-400/20 to-transparent" />

  <div className="relative mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-5 py-4">
    <div className="flex items-center gap-3">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-green-700 via-emerald-500 to-lime-400 text-lg font-black text-white shadow-[0_0_28px_rgba(34,197,94,0.35)]">
        HL
      </div>

      <div>
        <div className="text-sm font-black text-green-900">Hauler Load Analytics</div>
        <div className="text-xs font-semibold text-slate-600">Two-Stage Hauler Load Detection</div>
      </div>
    </div>

    <div className="hidden items-center gap-2 rounded-2xl border border-white/50 bg-white/45 p-1 shadow-inner backdrop-blur-xl md:flex">
      <button
        onClick={() => setPageMode("LIVE")}
        className={`rounded-xl px-4 py-2 text-sm font-black transition ${
          pageMode === "LIVE"
            ? "bg-white/90 text-green-700 shadow-sm ring-1 ring-green-200/70"
            : "text-slate-500 hover:bg-white/50 hover:text-slate-900"
        }`}
      >
        Live CCTV
      </button>

      <button
        onClick={() => setPageMode("VALIDATION")}
        className={`rounded-xl px-4 py-2 text-sm font-black transition ${
          pageMode === "VALIDATION"
            ? "bg-white/90 text-green-700 shadow-sm ring-1 ring-green-200/70"
            : "text-slate-500 hover:bg-white/50 hover:text-slate-900"
        }`}
      >
        Validasi Evidence
        {pendingDetections.length ? (
          <span className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-700">
            {pendingDetections.length}
          </span>
        ) : null}
      </button>
    </div>
  </div>
</header>

      <main className="mx-auto max-w-[1500px] px-5 py-6">
        <div className="mb-5 grid gap-4 md:hidden">
          <div className="grid grid-cols-2 gap-2 rounded-3xl border border-white/70 bg-white/60 p-2 shadow-[0_14px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl">
            <button
              onClick={() => setPageMode("LIVE")}
              className={`rounded-2xl px-4 py-3 text-sm font-black transition ${pageMode === "LIVE" ? "bg-gradient-to-br from-green-700 via-emerald-600 to-lime-500 text-white shadow-[0_14px_34px_rgba(22,163,74,0.25)]" : "bg-white/55 text-slate-500 hover:bg-white hover:text-green-800"}`}
            >
              Live CCTV
            </button>
            <button
              onClick={() => setPageMode("VALIDATION")}
              className={`rounded-2xl px-4 py-3 text-sm font-black transition ${pageMode === "VALIDATION" ? "bg-gradient-to-br from-green-700 via-emerald-600 to-lime-500 text-white shadow-[0_14px_34px_rgba(22,163,74,0.25)]" : "bg-white/55 text-slate-500 hover:bg-white hover:text-green-800"}`}
            >
              Validasi ({pendingDetections.length})
            </button>
          </div>
        </div>

        {pageMode === "LIVE" ? (
          <div className="grid items-start gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
            <aside className="space-y-5 xl:sticky xl:top-24 xl:self-start">
              <section className="rounded-[28px] border border-white/70 bg-white/75 p-5 shadow-[0_18px_55px_rgba(15,23,42,0.07)] backdrop-blur-xl">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-base font-black text-slate-900">Sumber CCTV</h2>
                  </div>
                  <span className="rounded-full bg-green-50 px-3 py-1 text-[11px] font-black text-green-700">LIVE</span>
                </div>

                <div className="mt-5 space-y-2">
                  {cctvList.map((cctv) => (
                    <button
                      key={cctv}
                      onClick={() => setSelectedCctv(cctv)}
                      className={`w-full rounded-2xl px-4 py-3 text-left text-sm font-black transition ${
                        selectedCctv === cctv ? "bg-gradient-to-br from-green-700 via-emerald-600 to-lime-500 text-white shadow-[0_14px_34px_rgba(22,163,74,0.23)]" : "border border-white/70 bg-white/55 text-slate-600 hover:bg-green-50/80 hover:text-green-800"
                      }`}
                    >
                      {cctv}
                    </button>
                  ))}
                </div>

                <div className="mt-5 rounded-2xl border border-white/70 bg-white/55 p-4 text-xs text-slate-500 shadow-inner backdrop-blur-xl">
                  <div className="font-black text-slate-700">Input aktif</div>
                  <div className="mt-1 break-all font-mono">{VIDEO_SOURCE}</div>
                </div>
              </section>

              <section className="rounded-[28px] border border-white/70 bg-white/75 p-5 shadow-[0_18px_55px_rgba(15,23,42,0.07)] backdrop-blur-xl">
                <h2 className="text-base font-black text-slate-900">Latency Monitor</h2>

                <div className="mt-5 grid gap-3">
                  <MiniMetric label="Detector" value={lastDetectorLatency == null ? "-" : `${lastDetectorLatency} ms`} />
                  <MiniMetric label="Classifier" value={lastClassifierLatency == null ? "-" : `${lastClassifierLatency} ms`} />
                  <MiniMetric label="Evidence Round" value={lastRoundTripLatency == null ? "-" : `${lastRoundTripLatency} ms`} />
                </div>
              </section>

              <section className="rounded-[28px] border border-white/70 bg-white/75 p-5 shadow-[0_18px_55px_rgba(15,23,42,0.07)] backdrop-blur-xl">
                <h2 className="text-base font-black text-slate-900">Validation KPI</h2>

                <div className="mt-5 grid grid-cols-2 gap-3">
                  <StatCard label="Pending" value={`${summary.pending}`} sub="menunggu" />
                  <StatCard label="Total" value={`${summary.total}`} sub="tersimpan" />
                  <StatCard label="Valid" value={`${summary.valid}`} sub="sesuai" />
                  <StatCard label="Agree" value={`${summary.agreement}%`} sub="valid / total" />
                </div>
              </section>
            </aside>

            <section className="space-y-6">
              <section className="rounded-[32px] border border-white/70 bg-white/78 p-5 shadow-[0_20px_65px_rgba(15,23,42,0.08)] backdrop-blur-xl">
                <div className="mb-5 flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
                  <div>
                    <h1 className="text-2xl font-black text-slate-950 md:text-3xl">Live CCTV Hauler Detection</h1>
                    <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
                      Detector membaca frame kecil agar latency lebih rendah. Screenshot evidence hanya dibuat ketika hauler benar-benar terdeteksi sebagai event baru.
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {isScanning ? (
                      <button onClick={stopAutoDetection} className="rounded-2xl border border-white/30 bg-gradient-to-br from-red-500 via-rose-600 to-red-700 px-5 py-3 text-sm font-black text-white shadow-[0_16px_38px_rgba(220,38,38,0.22)] transition hover:-translate-y-0.5 hover:shadow-[0_18px_44px_rgba(220,38,38,0.30)]">
                        Stop Detection
                      </button>
                    ) : (
                      <button onClick={startAutoDetection} className="rounded-2xl border border-white/30 bg-gradient-to-br from-green-700 via-emerald-600 to-lime-500 px-5 py-3 text-sm font-black text-white shadow-[0_16px_38px_rgba(22,163,74,0.24)] transition hover:-translate-y-0.5 hover:shadow-[0_18px_44px_rgba(22,163,74,0.32)]">
                        Start Detection
                      </button>
                    )}

                    <button
                      onClick={() => setPageMode("VALIDATION")}
                      className="rounded-2xl border border-white/70 bg-white/60 px-5 py-3 text-sm font-black text-slate-700 shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white hover:text-green-800"
                    >
                      Buka Validasi ({pendingDetections.length})
                    </button>
                  </div>
                </div>

                <div className="overflow-hidden rounded-[28px] border border-slate-900/10 bg-slate-950/95 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_22px_70px_rgba(15,23,42,0.28)]">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3 px-1">
                    <div className="flex items-center gap-2">
                      <span className={`h-2.5 w-2.5 rounded-full ${isScanning ? "bg-green-400" : "bg-slate-500"}`} />
                      <span className="font-mono text-xs font-bold text-white">LIVE • {selectedCctv}</span>
                    </div>

                    <span className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-bold text-slate-200 shadow-inner backdrop-blur-xl">
                      {isProcessingDetection ? "Detector membaca frame..." : isClassifying ? "Classifier membaca crop..." : scanMessage}
                    </span>
                  </div>

                  <div className="relative overflow-hidden rounded-2xl bg-black">
                    <video
                      ref={videoRef}
                      src={VIDEO_SOURCE}
                      className="block w-full object-contain"
                      muted
                      loop
                      playsInline
                      controls
                      autoPlay
                      preload="auto"
                      onLoadedMetadata={(event) => {
                        const video = event.currentTarget
                        setFrameInfo({ width: video.videoWidth || 1280, height: video.videoHeight || 720 })
                      }}
                      onCanPlay={() => {
                        if (!autoStartedRef.current) {
                          autoStartedRef.current = true
                          startAutoDetection()
                        }
                      }}
                      onError={() => setError("Video tidak terbaca")}
                    />

                    <div className="pointer-events-none absolute left-4 top-4 rounded-xl bg-black/65 px-3 py-2 font-mono text-xs text-white backdrop-blur">
                      {latestResult ? formatVideoTime(latestResult.videoTime) : "00:00"}
                    </div>

                    {overlayBox && latestHaulerBox ? (
                      <div
                        className="pointer-events-none absolute rounded-sm border-[3px] border-cyan-400"
                        style={{
                          left: `${Math.max(0, overlayBox.left)}%`,
                          top: `${Math.max(0, overlayBox.top)}%`,
                          width: `${Math.min(100, overlayBox.width)}%`,
                          height: `${Math.min(100, overlayBox.height)}%`,
                          boxShadow: "0 0 0 1px rgba(255,255,255,.55), 0 0 28px rgba(34,211,238,.6)",
                        }}
                      >
                        <div className="absolute -top-9 left-0 whitespace-nowrap rounded-lg bg-cyan-500 px-3 py-1.5 text-[11px] font-black text-white shadow">
                          HAULER {toPercent(latestHaulerBox.confidence).toFixed(1)}%
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>

                {error ? <div className="mt-5 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-700">{error}</div> : null}
              </section>

              {latestResult && latestStyle ? (
                <section className="rounded-[32px] border border-white/70 bg-white/78 p-5 shadow-[0_20px_65px_rgba(15,23,42,0.08)] backdrop-blur-xl">
                  <div className="mb-4 flex flex-col justify-between gap-3 md:flex-row md:items-center">
                    <div>
                      <div className="text-xs font-black uppercase tracking-wide text-slate-400">Evidence terakhir</div>
                      <h2 className={`mt-1 text-xl font-black ${latestStyle.title}`}>{formatStatus(latestResult.status)}</h2>
                      <p className="mt-1 text-sm font-semibold text-slate-500">
                        Classifier {latestResult.confidence.toFixed(1)}% • Detector {latestResult.detectorConfidence.toFixed(1)}% • {formatVideoTime(latestResult.videoTime)}
                      </p>
                    </div>

                    <span className={`w-fit rounded-full border px-4 py-2 text-xs font-black ${latestStyle.chip}`}>{summary.pending} pending validation</span>
                  </div>

                  <div className="grid gap-4 md:grid-cols-[1.6fr_1fr]">
                    <button
                      onClick={() => setPreview({ title: "Full Screenshot", subtitle: latestResult.timestamp, imageUrl: lastSnapshot || latestResult.snapshotUrl, kind: "snapshot", item: latestResult })}
                      className="group overflow-hidden rounded-2xl border border-white/70 bg-white/60 p-2 text-left shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5 hover:border-green-300 hover:shadow-[0_18px_45px_rgba(22,163,74,0.12)]"
                    >
                      <img src={lastSnapshot || latestResult.snapshotUrl} alt="Frame terakhir" className="max-h-72 w-full rounded-xl object-contain" />
                      <div className="px-2 py-2 text-xs font-black text-slate-500 group-hover:text-green-700">Klik untuk perbesar screenshot</div>
                    </button>

                    <button
                      onClick={() => setPreview({ title: "Crop Hauler", subtitle: latestResult.timestamp, imageUrl: lastCrop || latestResult.cropUrl, kind: "crop", item: latestResult })}
                      className="group overflow-hidden rounded-2xl border border-white/70 bg-white/60 p-2 text-left shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5 hover:border-green-300 hover:shadow-[0_18px_45px_rgba(22,163,74,0.12)]"
                    >
                      <img src={lastCrop || latestResult.cropUrl} alt="Crop hauler" className="max-h-72 w-full rounded-xl object-contain" />
                      <div className="px-2 py-2 text-xs font-black text-slate-500 group-hover:text-green-700">Klik untuk perbesar crop</div>
                    </button>
                  </div>
                </section>
              ) : null}
            </section>
          </div>
        ) : (
          <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
            <section className="rounded-[32px] border border-white/70 bg-white/78 p-5 shadow-[0_20px_65px_rgba(15,23,42,0.08)] backdrop-blur-xl">
              <div className="sticky top-20 z-20 -mx-5 -mt-5 mb-5 flex flex-col justify-between gap-4 border-b border-white/70 bg-white/70 px-5 py-5 backdrop-blur-2xl md:flex-row md:items-center">
                <div>
                  <h1 className="text-2xl font-black text-slate-950 md:text-3xl">Validasi Evidence</h1>
                </div>

                <button onClick={() => setPageMode("LIVE")} className="w-fit rounded-2xl border border-white/20 bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900 px-5 py-3 text-sm font-black text-white shadow-[0_16px_38px_rgba(15,23,42,0.22)] transition hover:-translate-y-0.5">
                  Kembali ke Live CCTV
                </button>
              </div>

              <div className="grid gap-4">
                {pendingDetections.length ? (
                  pendingDetections.map((item) => {
                    const style = getStatusStyle(item.status)
                    const selectedCorrection = correctionStatus[item.id] || item.status

                    return (
                      <article key={item.id} className="rounded-[28px] border border-white/70 bg-white/50 p-4 shadow-[0_18px_55px_rgba(15,23,42,0.06)] backdrop-blur-xl">
                        <div className="grid gap-5 lg:grid-cols-[360px_minmax(0,1fr)]">
                          <div className="grid gap-3">
                            <button
                              onClick={() => setPreview({ title: "Full Screenshot", subtitle: item.timestamp, imageUrl: item.snapshotUrl, kind: "snapshot", item })}
                              className="group overflow-hidden rounded-2xl border border-white/70 bg-white/70 p-2 text-left shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5 hover:border-green-300 hover:shadow-[0_18px_45px_rgba(22,163,74,0.12)]"
                            >
                              <img src={item.snapshotUrl} alt="Bukti frame" className="h-52 w-full rounded-xl object-cover" />
                              <div className="px-2 py-2 text-xs font-black text-slate-500 group-hover:text-green-700">Full screenshot • klik perbesar</div>
                            </button>

                            <button
                              onClick={() => setPreview({ title: "Crop Hauler", subtitle: item.timestamp, imageUrl: item.cropUrl, kind: "crop", item })}
                              className="group overflow-hidden rounded-2xl border border-white/70 bg-white/70 p-2 text-left shadow-sm backdrop-blur-xl transition hover:-translate-y-0.5 hover:border-green-300 hover:shadow-[0_18px_45px_rgba(22,163,74,0.12)]"
                            >
                              <img src={item.cropUrl} alt="Crop hauler" className="h-40 w-full rounded-xl object-cover" />
                              <div className="px-2 py-2 text-xs font-black text-slate-500 group-hover:text-green-700">Crop hauler • klik perbesar</div>
                            </button>
                          </div>

                          <div className="rounded-3xl border border-white/70 bg-white/75 p-5 shadow-inner backdrop-blur-xl">
                            <div className="flex flex-col justify-between gap-3 md:flex-row md:items-start">
                              <div>
                                <div className="text-xs font-black uppercase tracking-wide text-slate-400">Prediksi Model</div>
                                <div className={`mt-1 text-3xl font-black ${style.title}`}>{formatStatus(item.status)}</div>
                                <div className="mt-2 text-sm font-semibold text-slate-500">
                                  Classifier {item.confidence.toFixed(1)}% • Detector {item.detectorConfidence.toFixed(1)}% • {formatVideoTime(item.videoTime)}
                                </div>
                              </div>

                              <span className={`w-fit rounded-full border px-4 py-2 text-xs font-black ${style.chip}`}>PENDING</span>
                            </div>

                            <div className="mt-5 grid gap-3 md:grid-cols-3">
                              <MiniMetric label="Detector latency" value={`${item.detectorLatencyMs} ms`} />
                              <MiniMetric label="Classifier latency" value={`${item.classifierLatencyMs} ms`} />
                              <MiniMetric label="Total object" value={`${item.totalDetection}`} />
                            </div>

                            <div className="mt-5 rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">
                              <b>Analisis:</b> {item.notes}
                            </div>

                            <div className="mt-5 text-xs font-black uppercase tracking-wide text-slate-400">Koreksi label jika ML salah</div>
                            <div className="mt-2 grid gap-2 sm:grid-cols-3">
                              {statusOptions.map((status) => (
                                <button
                                  key={`${item.id}-${status}`}
                                  onClick={() => setCorrectionStatus((prev) => ({ ...prev, [item.id]: status }))}
                                  className={`rounded-2xl border px-3 py-3 text-sm font-black transition hover:-translate-y-0.5 ${
                                    selectedCorrection === status ? `${getStatusStyle(status).chip} shadow-sm` : "border-white/70 bg-white/65 text-slate-500 hover:bg-white hover:text-green-800"
                                  }`}
                                >
                                  {formatStatus(status)}
                                </button>
                              ))}
                            </div>

                            <textarea
                              value={validatorNotes[item.id] || ""}
                              onChange={(event) => setValidatorNotes((prev) => ({ ...prev, [item.id]: event.target.value }))}
                              placeholder="Catatan validasi opsional."
                              className="mt-4 h-20 w-full resize-none rounded-2xl border border-white/70 bg-white/75 p-4 text-sm outline-none shadow-inner backdrop-blur-xl transition focus:border-green-500 focus:ring-4 focus:ring-green-100"
                            />

                            <div className="mt-4 grid gap-3 sm:grid-cols-2">
                              <button onClick={() => validateDetection(item.id, "VALID")} className="rounded-2xl border border-white/30 bg-gradient-to-br from-green-700 via-emerald-600 to-lime-500 px-4 py-4 text-sm font-black text-white shadow-[0_16px_38px_rgba(22,163,74,0.22)] transition hover:-translate-y-0.5">
                                Valid
                              </button>
                              <button onClick={() => validateDetection(item.id, "INVALID")} className="rounded-2xl border border-white/30 bg-gradient-to-br from-red-500 via-rose-600 to-red-700 px-4 py-4 text-sm font-black text-white shadow-[0_16px_38px_rgba(220,38,38,0.20)] transition hover:-translate-y-0.5">
                                Tidak Valid
                              </button>
                            </div>
                          </div>
                        </div>
                      </article>
                    )
                  })
                ) : (
                  <div className="rounded-[28px] border border-dashed border-slate-300 bg-slate-50 p-12 text-center">
                    <div className="text-lg font-black text-slate-700">Belum ada evidence pending</div>
                    <p className="mt-2 text-sm text-slate-500">Evidence akan muncul ketika Model 1 mendeteksi hauler dan Model 2 selesai membaca crop muatan.</p>
                  </div>
                )}
              </div>
            </section>

            <aside className="space-y-5 xl:sticky xl:top-24 xl:self-start">
              <section className="rounded-[32px] border border-white/70 bg-white/78 p-5 shadow-[0_20px_65px_rgba(15,23,42,0.08)] backdrop-blur-xl">
                <h2 className="font-black text-slate-900">Ringkasan</h2>
                <div className="mt-5 grid gap-3">
                  <StatCard label="Pending" value={`${summary.pending}`} sub="butuh validasi" />
                  <StatCard label="Total Validasi" value={`${summary.total}`} sub="sudah diproses" />
                  <StatCard label="Valid" value={`${summary.valid}`} sub="model sesuai" />
                  <StatCard label="Invalid" value={`${summary.invalid}`} sub="model dikoreksi" />
                  <StatCard label="Agreement" value={`${summary.agreement}%`} sub="akurasi validasi manual" />
                </div>
              </section>

              <section className="rounded-[32px] border border-white/70 bg-white/78 p-5 shadow-[0_20px_65px_rgba(15,23,42,0.08)] backdrop-blur-xl">
                <div className="flex items-center justify-between gap-3">
                  <h2 className="font-black text-slate-900">Riwayat</h2>
                  <select
                    value={filterStatus}
                    onChange={(event) => setFilterStatus(event.target.value as "ALL" | Status)}
                    className="rounded-2xl border border-white/70 bg-white/70 px-3 py-2 text-xs font-bold outline-none shadow-inner backdrop-blur-xl"
                  >
                    <option value="ALL">Semua</option>
                    <option value="UNDERLOAD">Under</option>
                    <option value="IDEAL">Ideal</option>
                    <option value="OVERLOAD">Over</option>
                  </select>
                </div>

                <div className="mt-5 max-h-[640px] space-y-3 overflow-y-auto pr-1">
                  {filteredHistory.length ? (
                    filteredHistory.map((item) => {
                      const finalStatus = item.validatedAs || item.status
                      const style = getStatusStyle(finalStatus)

                      return (
                        <div
                          key={item.id}
                          className={`rounded-2xl border p-4 shadow-sm ${item.validation === "INVALID" ? "border-red-200 bg-red-50" : "border-slate-200 bg-white"}`}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <div className={`text-sm font-black ${style.title}`}>{formatStatus(finalStatus)}</div>
                              <div className="mt-1 text-xs font-semibold text-slate-500">{item.cctv}</div>
                            </div>

                            <span
                              className={`rounded-full border px-2 py-1 text-[11px] font-black ${
                                item.validation === "VALID" ? "border-green-200 bg-green-50 text-green-700" : "border-red-200 bg-red-50 text-red-700"
                              }`}
                            >
                              {item.validation}
                            </span>
                          </div>

                          <div className="mt-3 grid grid-cols-2 gap-2">
                            <img src={item.snapshotUrl} alt="Bukti frame" className="h-24 w-full rounded-xl border border-slate-200 object-cover" />
                            <img src={item.cropUrl} alt="Crop hauler" className="h-24 w-full rounded-xl border border-slate-200 object-cover" />
                          </div>

                          <div className="mt-3 text-xs leading-5 text-slate-500">
                            {item.timestamp} • {formatVideoTime(item.videoTime)} • classifier {item.confidence.toFixed(1)}%
                          </div>

                          {item.validatorNote ? <div className="mt-2 text-xs text-slate-600">“{item.validatorNote}”</div> : null}
                        </div>
                      )
                    })
                  ) : (
                    <div className="rounded-2xl bg-slate-50 p-6 text-center text-sm text-slate-500">Belum ada riwayat validasi.</div>
                  )}
                </div>
              </section>
            </aside>
          </div>
        )}
      </main>

{preview ? (
  <div
    className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm"
    onClick={() => setPreview(null)}
  >
    <div
      className="flex h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-[32px] bg-white shadow-2xl"
      onClick={(event) => event.stopPropagation()}
    >
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between gap-4 border-b border-slate-200 px-6 py-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <div className="truncate text-lg font-black text-slate-900">
              {preview.title}
            </div>

            <span className="rounded-full bg-slate-100 px-3 py-1 text-[11px] font-black uppercase tracking-wide text-slate-500">
              {preview.kind === "crop" ? "Crop Hauler" : "Full Evidence"}
            </span>
          </div>

          <div className="mt-1 truncate text-xs font-semibold text-slate-500">
            {preview.subtitle}
          </div>
        </div>

        <button
          onClick={() => setPreview(null)}
          className="shrink-0 rounded-2xl bg-slate-100 px-5 py-2.5 text-sm font-black text-slate-700 transition hover:bg-slate-200"
        >
          Tutup
        </button>
      </div>

      {/* Image Preview */}
      <div className="flex min-h-0 flex-1 items-center justify-center bg-slate-950 px-6 py-5">
        <div className="flex h-full w-full items-center justify-center rounded-[28px] border border-white/10 bg-white/5 p-4">
          <img
            src={preview.imageUrl}
            alt={preview.title}
            className={
              preview.kind === "crop"
                ? "block w-[50%] max-w-[720px] rounded-2xl object-contain shadow-2xl"
                : "block max-h-full max-w-full rounded-2xl object-contain shadow-2xl"
            }
          />
        </div>
      </div>

      {/* Metrics */}
      <div className="shrink-0 border-t border-slate-200 bg-white px-6 py-4">
        <div className="grid gap-3 md:grid-cols-4">
          <MiniMetric label="Prediksi" value={formatStatus(preview.item.status)} />
          <MiniMetric label="Classifier" value={`${preview.item.confidence.toFixed(1)}%`} />
          <MiniMetric label="Detector" value={`${preview.item.detectorConfidence.toFixed(1)}%`} />
          <MiniMetric label="Video Time" value={formatVideoTime(preview.item.videoTime)} />
        </div>
      </div>
    </div>
  </div>
) : null}
    </div>
  )
}
