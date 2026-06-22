import Foundation

/// AW139 ATA chapters and training-material categories.
/// Ported from `client/src/data/ata-codes.ts` and the DiagnosticForm select options
/// so the offline iPad app offers the same choices as the web platform.
enum ATACatalog {

    struct Entry: Identifiable, Hashable {
        let code: String
        let system: String
        var id: String { code }
        var label: String { "ATA \(code) - \(system)" }
    }

    /// Non-ATA training materials (engine / avionics / airframe type training).
    static let trainingMaterials: [Entry] = [
        Entry(code: "PRIMUS_EPIC", system: "Primus Epic (Avionics System)"),
        Entry(code: "PT6C_67CD", system: "PT6C-67CD (Engine)"),
        Entry(code: "AW139_AIRFRAME", system: "AW139 Airframe Type Training")
    ]

    static let ataCodes: [Entry] = [
        Entry(code: "05", system: "Periodic Inspection"),
        Entry(code: "10", system: "General"),
        Entry(code: "11", system: "Placards and Signs"),
        Entry(code: "12", system: "Servicing"),
        Entry(code: "13", system: "Avionics"),
        Entry(code: "14", system: "Special Procedures"),
        Entry(code: "20", system: "Standard Practices"),
        Entry(code: "21", system: "Air Conditioning"),
        Entry(code: "22", system: "Auto Flight"),
        Entry(code: "23", system: "Communications"),
        Entry(code: "24", system: "Electrical Power"),
        Entry(code: "25", system: "Equipment and Furnishings"),
        Entry(code: "26", system: "Fire Protection"),
        Entry(code: "27", system: "Flight Controls"),
        Entry(code: "28", system: "Fuel"),
        Entry(code: "29", system: "Hydraulic Power"),
        Entry(code: "30", system: "Ice and Rain Protection"),
        Entry(code: "31", system: "Indicating and Recorders"),
        Entry(code: "32", system: "Landing Gear"),
        Entry(code: "33", system: "Lighting"),
        Entry(code: "34", system: "Navigation"),
        Entry(code: "35", system: "Oxygen"),
        Entry(code: "36", system: "Pneumatic"),
        Entry(code: "37", system: "Vacuum"),
        Entry(code: "38", system: "Water/Waste"),
        Entry(code: "39", system: "Electrical Auxiliary Power"),
        Entry(code: "41", system: "Power Plant"),
        Entry(code: "42", system: "Engine Electrical"),
        Entry(code: "43", system: "Engine Fuel and Control"),
        Entry(code: "44", system: "Propulsion Nozzle"),
        Entry(code: "45", system: "Air Induction"),
        Entry(code: "46", system: "Exhaust"),
        Entry(code: "47", system: "Engine Cooling"),
        Entry(code: "49", system: "Auxiliary Power Unit"),
        Entry(code: "51", system: "Structures General"),
        Entry(code: "52", system: "Wings"),
        Entry(code: "53", system: "Fuselage"),
        Entry(code: "54", system: "Nacelles/Pylons"),
        Entry(code: "55", system: "Stabilizers"),
        Entry(code: "56", system: "Windows"),
        Entry(code: "57", system: "Wings/Movable Surfaces"),
        Entry(code: "71", system: "Powerplant"),
        Entry(code: "72", system: "Engine Installation"),
        Entry(code: "73", system: "Engine Fuel System"),
        Entry(code: "74", system: "Ignition System"),
        Entry(code: "75", system: "Air Induction System"),
        Entry(code: "76", system: "Engine Controls"),
        Entry(code: "77", system: "Engine Indicating System"),
        Entry(code: "78", system: "Exhaust System"),
        Entry(code: "80", system: "Starting System"),
        Entry(code: "81", system: "Generator System"),
        Entry(code: "82", system: "Electrical System"),
        Entry(code: "83", system: "Hydraulic System"),
        Entry(code: "84", system: "Pneumatic System"),
        Entry(code: "85", system: "Vacuum System"),
        Entry(code: "86", system: "Fuel System"),
        Entry(code: "87", system: "Oxygen System"),
        Entry(code: "88", system: "Water Injection System"),
        Entry(code: "89", system: "Propeller System")
    ]

    static func displayName(for code: String) -> String {
        if let t = trainingMaterials.first(where: { $0.code == code }) { return t.system }
        if let a = ataCodes.first(where: { $0.code == code }) { return a.label }
        return code
    }
}

/// Mirrors the `taskType` field from the web DiagnosticForm.
enum TaskType: String, CaseIterable, Identifiable {
    case faultIsolation = "fault_isolation"
    case procedure = "procedure"
    case inspection = "inspection"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .faultIsolation: return "Fault Isolation"
        case .procedure: return "Procedure / Task"
        case .inspection: return "Inspection"
        }
    }
}
