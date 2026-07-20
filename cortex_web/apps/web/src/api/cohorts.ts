import { authedFetch } from "./core";

// Cohort access is enforced by the server. Internal account identifiers never
// cross this boundary; managers address members by their nine-digit public ID.
export interface CohortSummary {
  cohortId: string;
  name: string;
  role: "manager" | "member";
  status: "invited" | "active";
  memberCount: number;
  createdUtc: string;
}

export interface CohortMemberInfo {
  publicId: string;
  status: "invited" | "active";
  isManager: boolean;
  isYou: boolean;
  joinedUtc: string | null;
  displayName?: string;
}

export interface CohortDetail {
  cohortId: string;
  name: string;
  role: "manager" | "member";
  status: "invited" | "active";
  createdUtc: string;
  members?: CohortMemberInfo[];
  emailInvites?: { email: string; invitedUtc: string }[];
}

export interface CohortPoint {
  ts: string;
  skill: number | null;
  bias: number | null;
  sd: number | null;
  phase: string;
}

export interface CohortTaskSeries {
  taskK: number;
  points: CohortPoint[];
}

export interface CohortMemberSeries extends Omit<CohortMemberInfo, "status"> {
  status?: string;
  series: CohortTaskSeries[];
}

export interface CohortPerformance {
  cohortId: string;
  name: string;
  from: string;
  to: string;
  ellStar?: number[] | null;
  members: CohortMemberSeries[];
}

export function listCohorts(): Promise<{ cohorts: CohortSummary[] }> {
  return authedFetch("/api/cohorts", {}, { retries: 2 });
}

export function createCohort(name: string): Promise<{ cohortId: string; name: string }> {
  return authedFetch("/api/cohorts", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function getCohort(cohortId: string): Promise<CohortDetail> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}`, {}, { retries: 2 });
}

// days <= 0 selects all time; omission preserves the server default.
export function getCohortPerformance(cohortId: string, days?: number): Promise<CohortPerformance> {
  const query = days == null ? "" : `?days=${days}`;
  return authedFetch(
    `/api/cohorts/${encodeURIComponent(cohortId)}/performance${query}`,
    {},
    { retries: 2 },
  );
}

export function inviteToCohort(cohortId: string, publicId: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/invite`, {
    method: "POST",
    body: JSON.stringify({ publicId }),
  });
}

// This endpoint is intentionally anti-oracle: unknown addresses receive an
// invitation without revealing whether an account exists.
export function inviteToCohortByEmail(cohortId: string, email: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/invite-email`, {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function cancelCohortEmailInvite(cohortId: string, email: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/invite-email/cancel`, {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function acceptCohortInvite(cohortId: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/accept`, { method: "POST" });
}

export function declineCohortInvite(cohortId: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/decline`, { method: "POST" });
}

export function leaveCohort(cohortId: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/leave`, { method: "POST" });
}

export function removeCohortMember(cohortId: string, publicId: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}/remove`, {
    method: "POST",
    body: JSON.stringify({ publicId }),
  });
}

export function deleteCohort(cohortId: string): Promise<{ ok: boolean }> {
  return authedFetch(`/api/cohorts/${encodeURIComponent(cohortId)}`, { method: "DELETE" });
}
