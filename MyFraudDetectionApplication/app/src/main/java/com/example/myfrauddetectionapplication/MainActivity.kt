package com.example.myfrauddetectionapplication

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

// ── Palette ──────────────────────────────────────────────────────────────────
private val BgPage    = Color(0xFFF4EFE6)
private val BgCard    = Color(0xFFFFFAF2)
private val Ink       = Color(0xFF1E1A16)
private val Accent    = Color(0xFF1F6F78)
private val Accent2   = Color(0xFFD96C3F)
private val Border    = Color(0xFFD9CCB8)
private val Muted     = Color(0xFF6F655C)
private val FraudBg   = Color(0xFFFDE2DF)
private val FraudFg   = Color(0xFFB63C31)
private val OkBg      = Color(0xFFDEF3E3)
private val OkFg      = Color(0xFF2D7D46)
private val WarnBg    = Color(0xFFFFF0CF)
private val WarnFg    = Color(0xFFB17A16)

// ── Data classes ─────────────────────────────────────────────────────────────
data class AnalysisResult(
    @SerializedName("predicted_class") val predictedClass: String?,
    @SerializedName("fraud_probability") val fraudProbability: Double?,
    @SerializedName("risk_level") val riskLevel: String?,
    @SerializedName("recommendation") val recommendation: String?,
    @SerializedName("markers") val markers: List<String>?,
    @SerializedName("decision_reasons") val decisionReasons: List<String>?,
    @SerializedName("features") val features: Features?,
    @SerializedName("suspicious_segments") val suspiciousSegments: List<Segment>?
)
data class Features(
    @SerializedName("scenario_type") val scenarioType: String?,
    @SerializedName("channel") val channel: String?
)
data class Segment(@SerializedName("text") val text: String?)

// ── HTTP client (singleton) ───────────────────────────────────────────────────
private val httpClient = OkHttpClient.Builder()
    .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
    .readTimeout(120, java.util.concurrent.TimeUnit.SECONDS)
    .writeTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
    .build()
private val gson = Gson()

// ── Activity ──────────────────────────────────────────────────────────────────
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                FraudAnalyzerApp()
            }
        }
    }
}

// ── Root composable ───────────────────────────────────────────────────────────
@Composable
fun FraudAnalyzerApp() {
    val context = LocalContext.current
    val scope   = rememberCoroutineScope()

    // State
    var serverUrl       by remember { mutableStateOf("http://10.0.2.2:5000") }
    var selectedTab     by remember { mutableIntStateOf(0) }
    var transcript      by remember { mutableStateOf("") }
    var selectedUri     by remember { mutableStateOf<Uri?>(null) }
    var selectedName    by remember { mutableStateOf("Нажми для выбора аудиофайла") }
    var whisperText     by remember { mutableStateOf("") }
    var showTranscript  by remember { mutableStateOf(false) }
    var showAnalyzeBtn  by remember { mutableStateOf(false) }
    var isTranscribing  by remember { mutableStateOf(false) }
    var isAnalyzing     by remember { mutableStateOf(false) }
    var progressText    by remember { mutableStateOf("") }
    var result          by remember { mutableStateOf<AnalysisResult?>(null) }
    var errorMsg        by remember { mutableStateOf<String?>(null) }

    // File picker launcher
    val fileLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            selectedUri = it
            // Get display name
            context.contentResolver.query(it, null, null, null, null)?.use { cursor ->
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst() && idx != -1) selectedName = cursor.getString(idx)
            }
            showTranscript = false
            showAnalyzeBtn = false
        }
    }

    // Permission launcher
    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) fileLauncher.launch("audio/*")
        else errorMsg = "Нужно разрешение для доступа к файлам"
    }

    fun pickFile() {
        val perm = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
            Manifest.permission.READ_MEDIA_AUDIO
        else Manifest.permission.READ_EXTERNAL_STORAGE
        if (ContextCompat.checkSelfPermission(context, perm) == PackageManager.PERMISSION_GRANTED)
            fileLauncher.launch("audio/*")
        else permLauncher.launch(perm)
    }

    fun transcribe() {
        val uri = selectedUri ?: run { errorMsg = "Сначала выберите аудиофайл"; return }
        val url = serverUrl.trimEnd('/')
        isTranscribing = true
        progressText = "Копирование файла..."
        scope.launch {
            try {
                val tmp = withContext(Dispatchers.IO) {
                    val name = selectedName
                    val ext  = name.substringAfterLast('.', "audio")
                    val f    = File.createTempFile("upload_", ".$ext", context.cacheDir)
                    context.contentResolver.openInputStream(uri)?.use { inp ->
                        FileOutputStream(f).use { inp.copyTo(it) }
                    }
                    f
                }
                progressText = "Whisper транскрибирует..."
                val text = withContext(Dispatchers.IO) {
                    val body = MultipartBody.Builder().setType(MultipartBody.FORM)
                        .addFormDataPart("audio", tmp.name, tmp.asRequestBody("audio/*".toMediaType()))
                        .build()
                    val req = Request.Builder().url("$url/transcribe").post(body).build()
                    httpClient.newCall(req).execute().use { resp ->
                        val json = resp.body?.string() ?: throw Exception("Пустой ответ")
                        if (!resp.isSuccessful) throw Exception(
                            runCatching { JSONObject(json).getString("error") }.getOrDefault(json)
                        )
                        JSONObject(json).getString("transcript")
                    }.also { tmp.delete() }
                }
                whisperText    = text
                showTranscript = true
                showAnalyzeBtn = true
            } catch (e: Exception) {
                errorMsg = "Ошибка транскрипции: ${e.message}"
            } finally {
                isTranscribing = false
                progressText   = ""
            }
        }
    }

    fun analyze(text: String) {
        val url = serverUrl.trimEnd('/')
        isAnalyzing = true
        result      = null
        scope.launch {
            try {
                val r = withContext(Dispatchers.IO) {
                    val body = JSONObject().put("transcript", text).toString()
                        .toRequestBody("application/json".toMediaType())
                    val req = Request.Builder().url("$url/analyze-call").post(body).build()
                    httpClient.newCall(req).execute().use { resp ->
                        val json = resp.body?.string() ?: throw Exception("Пустой ответ")
                        if (!resp.isSuccessful) throw Exception(
                            runCatching { JSONObject(json).getString("error") }.getOrDefault(json)
                        )
                        gson.fromJson(json, AnalysisResult::class.java)
                    }
                }
                result = r
            } catch (e: Exception) {
                errorMsg = "Ошибка анализа: ${e.message}"
            } finally {
                isAnalyzing = false
            }
        }
    }

    // ── UI ────────────────────────────────────────────────────────────────────
    Box(Modifier.fillMaxSize().background(BgPage)) {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
                .systemBarsPadding()
        ) {
            // Header
            Text("🔍 Fraud Call Analyzer", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = Ink)
            Text("Анализ звонков на мошенничество", fontSize = 14.sp, color = Muted,
                modifier = Modifier.padding(bottom = 16.dp))

            // Server URL card
            Card(modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                colors = CardDefaults.cardColors(containerColor = BgCard),
                shape = RoundedCornerShape(16.dp)) {
                Column(Modifier.padding(16.dp)) {
                    Text("АДРЕС СЕРВЕРА", fontSize = 11.sp, color = Muted, letterSpacing = 0.1.sp)
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(
                        value = serverUrl,
                        onValueChange = { serverUrl = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("http://192.168.1.100:8000") },
                        singleLine = true,
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Accent,
                            focusedLabelColor = Accent
                        )
                    )
                }
            }

            // Tabs
            val tabs = listOf("✏️ Транскрипт", "🎙 Аудио (Whisper)")
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = BgCard,
                contentColor = Accent,
                indicator = { tabPositions ->
                    Box(Modifier.tabIndicatorOffset(tabPositions[selectedTab])
                        .height(3.dp).background(Accent, RoundedCornerShape(topStart = 3.dp, topEnd = 3.dp)))
                },
                modifier = Modifier.clip(RoundedCornerShape(12.dp)).padding(bottom = 12.dp)
            ) {
                tabs.forEachIndexed { i, title ->
                    Tab(selected = selectedTab == i, onClick = { selectedTab = i },
                        text = { Text(title, fontSize = 14.sp,
                            color = if (selectedTab == i) Accent else Muted) })
                }
            }

            // ── Tab 0: Text ──
            AnimatedVisibility(selectedTab == 0) {
                Card(modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                    colors = CardDefaults.cardColors(containerColor = BgCard),
                    shape = RoundedCornerShape(16.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        OutlinedTextField(
                            value = transcript,
                            onValueChange = { transcript = it },
                            modifier = Modifier.fillMaxWidth().height(200.dp),
                            placeholder = { Text("Вставьте транскрипт звонка...", color = Muted) },
                            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Accent)
                        )
                        Spacer(Modifier.height(12.dp))
                        Button(
                            onClick = {
                                if (transcript.isBlank()) errorMsg = "Вставьте транскрипт"
                                else analyze(transcript)
                            },
                            enabled = !isAnalyzing,
                            colors = ButtonDefaults.buttonColors(containerColor = Accent),
                            shape = RoundedCornerShape(999.dp)
                        ) { Text(if (isAnalyzing) "Анализируем..." else "Анализировать") }
                    }
                }
            }

            // ── Tab 1: Audio ──
            AnimatedVisibility(selectedTab == 1) {
                Card(modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                    colors = CardDefaults.cardColors(containerColor = BgCard),
                    shape = RoundedCornerShape(16.dp)) {
                    Column(Modifier.padding(16.dp)) {

                        // Drop zone
                        Box(
                            Modifier.fillMaxWidth().height(110.dp)
                                .border(2.dp, Border, RoundedCornerShape(12.dp))
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.White)
                                .clickable { pickFile() },
                            contentAlignment = Alignment.Center
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("🎧", fontSize = 36.sp)
                                Text(selectedName, fontSize = 14.sp, color = Muted,
                                    textAlign = TextAlign.Center, modifier = Modifier.padding(top = 8.dp))
                            }
                        }

                        // Progress
                        AnimatedVisibility(isTranscribing) {
                            Column(Modifier.padding(top = 12.dp)) {
                                LinearProgressIndicator(modifier = Modifier.fillMaxWidth(),
                                    color = Accent, trackColor = Border)
                                Text(progressText, fontSize = 13.sp, color = Muted,
                                    modifier = Modifier.padding(top = 4.dp))
                            }
                        }

                        // Whisper transcript preview
                        AnimatedVisibility(showTranscript) {
                            Column(Modifier.padding(top = 12.dp)) {
                                Text("ТРАНСКРИПТ (можно редактировать)",
                                    fontSize = 11.sp, color = Muted, letterSpacing = 0.1.sp)
                                Spacer(Modifier.height(6.dp))
                                OutlinedTextField(
                                    value = whisperText,
                                    onValueChange = { whisperText = it },
                                    modifier = Modifier.fillMaxWidth().height(140.dp),
                                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Accent)
                                )
                            }
                        }

                        // Buttons row
                        Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = { transcribe() },
                                enabled = !isTranscribing,
                                colors = ButtonDefaults.buttonColors(containerColor = Accent),
                                shape = RoundedCornerShape(999.dp)
                            ) { Text(if (isTranscribing) "Транскрибируем..." else "🎙 Транскрибировать") }

                            AnimatedVisibility(showAnalyzeBtn) {
                                Button(
                                    onClick = {
                                        if (whisperText.isBlank()) errorMsg = "Транскрипт пуст"
                                        else analyze(whisperText)
                                    },
                                    enabled = !isAnalyzing,
                                    colors = ButtonDefaults.buttonColors(containerColor = Accent2),
                                    shape = RoundedCornerShape(999.dp)
                                ) { Text(if (isAnalyzing) "Анализируем..." else "Анализировать") }
                            }
                        }
                    }
                }
            }

            // ── Loading ──
            AnimatedVisibility(isAnalyzing) {
                Box(Modifier.fillMaxWidth().padding(24.dp), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Accent)
                        Text("Анализируем транскрипт...", color = Muted, fontSize = 14.sp,
                            modifier = Modifier.padding(top = 12.dp))
                    }
                }
            }

            // ── Result card ──
            result?.let { r -> ResultCard(r) }
        }

        // ── Error snackbar ──
        errorMsg?.let { msg ->
            Snackbar(
                modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
                action = { TextButton(onClick = { errorMsg = null }) { Text("OK", color = Accent) } },
                containerColor = Ink
            ) { Text(msg, color = Color.White) }
        }
    }
}

// ── Result card ───────────────────────────────────────────────────────────────
@Composable
fun ResultCard(r: AnalysisResult) {
    val cls = r.predictedClass?.lowercase() ?: "unknown"
    val (pillBg, pillFg) = when (cls) {
        "fraud"      -> FraudBg to FraudFg
        "suspicious" -> WarnBg  to WarnFg
        else         -> OkBg    to OkFg
    }

    Card(modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
        colors = CardDefaults.cardColors(containerColor = BgCard),
        shape = RoundedCornerShape(16.dp)) {
        Column(Modifier.padding(16.dp)) {

            // Verdict badge
            Box(
                Modifier.background(pillBg, RoundedCornerShape(999.dp))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text(cls.uppercase(), color = pillFg, fontWeight = FontWeight.Bold, fontSize = 18.sp)
            }

            Spacer(Modifier.height(16.dp))

            // Stats 2x2
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                StatBox("ВЕРОЯТНОСТЬ", "%.1f%%".format((r.fraudProbability ?: 0.0) * 100), Modifier.weight(1f))
                StatBox("УРОВЕНЬ РИСКА", r.riskLevel ?: "—", Modifier.weight(1f))
            }
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                StatBox("СЦЕНАРИЙ", r.features?.scenarioType ?: "—", Modifier.weight(1f))
                StatBox("КАНАЛ", r.features?.channel ?: "—", Modifier.weight(1f))
            }

            Spacer(Modifier.height(8.dp))
            InfoBox("РЕКОМЕНДАЦИЯ", r.recommendation ?: "—")
            Spacer(Modifier.height(8.dp))
            InfoBox("МАРКЕРЫ МОШЕННИЧЕСТВА",
                r.markers?.joinToString("\n") { "• $it" } ?: "—")
            Spacer(Modifier.height(8.dp))
            InfoBox("ПОЧЕМУ ТАК РЕШИЛА МОДЕЛЬ",
                r.decisionReasons?.joinToString("\n") { "• $it" } ?: "—")
            Spacer(Modifier.height(8.dp))

            // Suspicious segment
            Column(Modifier.fillMaxWidth().border(1.dp, Border, RoundedCornerShape(10.dp))
                .clip(RoundedCornerShape(10.dp)).background(Color.White).padding(12.dp)) {
                Text("ПОДОЗРИТЕЛЬНЫЙ ФРАГМЕНТ", fontSize = 10.sp, color = Muted, letterSpacing = 0.1.sp)
                Spacer(Modifier.height(6.dp))
                Text(
                    r.suspiciousSegments?.firstOrNull()?.text ?: "Подозрительный сегмент не найден",
                    fontSize = 13.sp, color = Ink, fontFamily = FontFamily.Monospace,
                    modifier = Modifier.fillMaxWidth().background(Color(0xFFF8F4EE), RoundedCornerShape(6.dp)).padding(10.dp)
                )
            }
        }
    }
}

@Composable
fun StatBox(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier.border(1.dp, Border, RoundedCornerShape(10.dp))
        .clip(RoundedCornerShape(10.dp)).background(Color.White).padding(12.dp)) {
        Text(label, fontSize = 10.sp, color = Muted, letterSpacing = 0.1.sp)
        Text(value, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Ink,
            modifier = Modifier.padding(top = 4.dp))
    }
}

@Composable
fun InfoBox(label: String, value: String) {
    Column(Modifier.fillMaxWidth().border(1.dp, Border, RoundedCornerShape(10.dp))
        .clip(RoundedCornerShape(10.dp)).background(Color.White).padding(12.dp)) {
        Text(label, fontSize = 10.sp, color = Muted, letterSpacing = 0.1.sp)
        Text(value, fontSize = 14.sp, color = Ink, modifier = Modifier.padding(top = 6.dp))
    }
}