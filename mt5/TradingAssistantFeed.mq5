//+------------------------------------------------------------------+
//| TradingAssistantFeed.mq5                                          |
//| Sends this chart's symbol (XAUUSD) candles and ticks from MT5 to  |
//| the Trading Assistant app, so the app's chart matches MT5 exactly.|
//|                                                                   |
//| Times are sent exactly as MT5 shows them (broker server time).    |
//+------------------------------------------------------------------+
#property copyright "Trading Assistant"
#property version   "1.10"
#property strict

input string ServerUrl     = "http://127.0.0.1:8000"; // must also be listed in Tools > Options > Expert Advisors > Allow WebRequest
input string Secret        = "";                       // same value as MT5_SECRET in the app's .env (leave empty if unused)
input int    HistoryBars   = 1500;                     // candles sent at start for Daily and above (the lower timeframes use the counts below)
input int    Bars5m        = 6000;                     // ~3 weeks
input int    Bars15m       = 60000;                    // ~2.5 years (the 15M is vital: entries are on it)
input int    Bars1h        = 18000;                    // ~3 years (the 1H is vital: entries are on it)
input int    Bars4h        = 3000;                     // higher timeframes need far fewer
input int    TickPushMs    = 250;                      // how often to look for a new tick
input int    CandleSyncSec = 2;                        // how often the newest candles are re-sent

#define TF_COUNT 7
ENUM_TIMEFRAMES g_tfs[TF_COUNT]   = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1, PERIOD_W1, PERIOD_MN1};
string          g_names[TF_COUNT] = {"5m", "15m", "1h", "4h", "1d", "1w", "1mo"};

int HistoryFor(const int idx)
  {
   switch(idx)
     {
      case 0: return(Bars5m);
      case 1: return(Bars15m);
      case 2: return(Bars1h);
      case 3: return(Bars4h);
     }
   return(HistoryBars);
  }

bool   g_needHistory = true;
double g_lastBid     = 0.0;
ulong  g_lastSyncMs  = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   if(!EventSetMillisecondTimer(MathMax(50, TickPushMs)))
      return(INIT_FAILED);
   PrintFormat("TradingAssistantFeed: sending %s to %s", _Symbol, ServerUrl);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//+------------------------------------------------------------------+
//| POST a JSON body. Returns the HTTP status code (-1 on failure).  |
//+------------------------------------------------------------------+
int Post(const string path, const string body, string &reply)
  {
   char   data[], result[];
   string headers = "Content-Type: application/json\r\n";
   if(Secret != "")
      headers += "X-Secret: " + Secret + "\r\n";

   int n = StringToCharArray(body, data, 0, StringLen(body), CP_UTF8);
   ArrayResize(data, n);

   string resultHeaders;
   ResetLastError();
   int code = WebRequest("POST", ServerUrl + path, headers, 5000, data, result, resultHeaders);
   if(code == -1)
     {
      PrintFormat("WebRequest failed (error %d). Add %s under Tools > Options > Expert Advisors > 'Allow WebRequest for listed URL'.",
                  GetLastError(), ServerUrl);
      reply = "";
      return(-1);
     }
   reply = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   if(code != 200)
      PrintFormat("Server replied HTTP %d: %s", code, reply);
   return(code);
  }

//+------------------------------------------------------------------+
//| JSON fragment  "1h":[[time,o,h,l,c],...]  for one timeframe      |
//+------------------------------------------------------------------+
string FrameJson(const int idx, const int count, const int start = 0)   // start = how many candles back from the newest to begin
  {
   MqlRates r[];
   ArraySetAsSeries(r, false);                 // oldest first
   int got = CopyRates(_Symbol, g_tfs[idx], start, count, r);
   if(got <= 0)
      return("");

   string s = "\"" + g_names[idx] + "\":[";
   for(int i = 0; i < got; i++)
     {
      if(i > 0)
         s += ",";
      s += "[" + IntegerToString((long)r[i].time) + ","
           + DoubleToString(r[i].open,  _Digits) + ","
           + DoubleToString(r[i].high,  _Digits) + ","
           + DoubleToString(r[i].low,   _Digits) + ","
           + DoubleToString(r[i].close, _Digits) + "]";
     }
   return(s + "]");
  }

//+------------------------------------------------------------------+
//| Send full history for one timeframe. true if the server took it. |
//+------------------------------------------------------------------+
bool SendHistory(const int idx)
  {
   // Long histories go in chunks of 4000 candles, oldest first (one huge request would be slow and could time out).
   const int CHUNK = 4000;
   int total = HistoryFor(idx);
   int avail = Bars(_Symbol, g_tfs[idx]);
   if(avail > 0 && avail < total)
      total = avail;
   if(total <= 0)
      return(false);
   bool any = false;
   for(int off = ((total - 1) / CHUNK) * CHUNK; off >= 0; off -= CHUNK)
     {
      string f = FrameJson(idx, MathMin(CHUNK, total - off), off);
      if(f == "")
         return(false);                        // terminal has not loaded this timeframe yet
      string reply;
      string body = "{\"symbol\":\"" + _Symbol + "\",\"frames\":{" + f + "}}";
      if(Post("/api/mt5/candles", body, reply) != 200)
         return(false);
      any = true;
     }
   return(any);
  }

//+------------------------------------------------------------------+
//| Re-send the newest 3 candles of every timeframe (covers the      |
//| forming candle and the one that just closed).                    |
//+------------------------------------------------------------------+
bool SendRecent(string &reply)
  {
   string body = "{\"symbol\":\"" + _Symbol + "\",\"frames\":{";
   bool   first = true;
   for(int i = 0; i < TF_COUNT; i++)
     {
      string f = FrameJson(i, 3);
      if(f == "")
         continue;
      if(!first)
         body += ",";
      body += f;
      first = false;
     }
   body += "}}";
   return(Post("/api/mt5/candles", body, reply) == 200);
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   // 1) latest tick
   MqlTick t;
   if(SymbolInfoTick(_Symbol, t) && t.bid > 0.0 && t.bid != g_lastBid)
     {
      string r;
      string body = "{\"t\":" + IntegerToString((long)t.time) + ",\"bid\":" + DoubleToString(t.bid, _Digits)
                    + ",\"ask\":" + DoubleToString(t.ask, _Digits) + "}";
      if(Post("/api/mt5/tick", body, r) == 200)
         g_lastBid = t.bid;
     }

   // 2) candles
   ulong now = GetTickCount64();
   if(g_needHistory)
     {
      bool all = true;
      for(int i = 0; i < TF_COUNT; i++)
         if(!SendHistory(i))
            all = false;
      g_needHistory = !all;                    // retry until every timeframe went through
      g_lastSyncMs  = now;
      return;
     }

   if(now - g_lastSyncMs >= (ulong)CandleSyncSec * 1000)
     {
      g_lastSyncMs = now;
      string reply;
      if(SendRecent(reply) && StringFind(reply, "\"need_history\":true") >= 0)
         g_needHistory = true;                 // the app restarted or lost data: send history again
     }
  }
//+------------------------------------------------------------------+
