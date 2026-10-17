# Reads AMD GPU telemetry through ADL's PMLog (atiadlxx.dll, installed with the
# Radeon driver): GFX clock, temperatures, power, activity, throttler status.
# Dot-source it, then call Read-AdlSensors. Sensor numbers are ADL_PMLOG_*.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Adl {
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] public delegate IntPtr Alloc(int size);
    [DllImport("atiadlxx.dll", CallingConvention = CallingConvention.Cdecl)] static extern int ADL2_Main_Control_Create(Alloc cb, int enumConnected, out IntPtr ctx);
    [DllImport("atiadlxx.dll", CallingConvention = CallingConvention.Cdecl)] static extern int ADL2_Adapter_NumberOfAdapters_Get(IntPtr ctx, out int n);
    [DllImport("atiadlxx.dll", CallingConvention = CallingConvention.Cdecl)] static extern int ADL2_New_QueryPMLogData_Get(IntPtr ctx, int adapter, IntPtr output);
    static readonly Alloc cb = size => Marshal.AllocHGlobal(size);
    static IntPtr ctx = IntPtr.Zero;
    const int Sensors = 256;
    const int Bytes = 4 + Sensors * 8;
    public static int Init() {
        if (ctx != IntPtr.Zero) return 0;
        return ADL2_Main_Control_Create(cb, 1, out ctx);
    }
    public static int AdapterCount() { int n; ADL2_Adapter_NumberOfAdapters_Get(ctx, out n); return n; }
    // Returns supported flag and value per sensor (2 ints each), or null on failure.
    public static int[] Read(int adapter) {
        IntPtr buf = Marshal.AllocHGlobal(Bytes);
        try {
            Marshal.WriteInt32(buf, 0, Bytes);
            for (int i = 4; i < Bytes; i += 4) Marshal.WriteInt32(buf, i, 0);
            int rc = ADL2_New_QueryPMLogData_Get(ctx, adapter, buf);
            if (rc != 0) return null;
            int[] o = new int[Sensors * 2];
            Marshal.Copy(IntPtr.Add(buf, 4), o, 0, Sensors * 2);
            return o;
        } finally { Marshal.FreeHGlobal(buf); }
    }
}
'@

# Sensors of interest (ADL_PMLOG_*): name -> index.
$script:AdlSensors = [ordered]@{
    gfx_clk_mhz = 1; mem_clk_mhz = 2; soc_clk_mhz = 3
    temp_edge_c = 8; temp_hotspot_c = 27; temp_gfx_c = 28; temp_soc_c = 29; temp_cpu_c = 32
    soc_power_w = 17; asic_power_w = 23; gfx_power_w = 30; cpu_power_w = 33; board_power_w = 73
    gfx_activity_pct = 19; mem_activity_pct = 20
    throttler_status = 35; throttle_power_pct = 55; throttle_temp_gfx_pct = 52
    apu_power_limit_w = 48; total_power_limit_w = 47
}

function Read-AdlSensors([int] $Adapter = -1) {
    if ([Adl]::Init() -ne 0) { throw 'ADL init failed' }
    $idx = if ($Adapter -ge 0) { @($Adapter) } else { 0..([Adl]::AdapterCount() - 1) }
    foreach ($a in $idx) {
        $raw = [Adl]::Read($a)
        if ($null -eq $raw) { continue }
        $o = [ordered]@{ adapter = $a }
        $any = $false
        foreach ($k in $script:AdlSensors.Keys) {
            $i = $script:AdlSensors[$k]
            if ($raw[2 * $i] -ne 0) { $o[$k] = $raw[2 * $i + 1]; $any = $true }
        }
        if ($any) { [pscustomobject]$o }
    }
}
