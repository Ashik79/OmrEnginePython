import os
import re

src_dir = r"d:\SohagPhysics web by asif\SohagPhysics\SohagPhysicsClient\src"

files_to_fix = [
    r"components\AddPayment.jsx",
    r"components\AddUsers.jsx",
    r"components\Admission.jsx",
    r"components\AttendanceBatch.jsx",
    r"components\BatchCard.jsx",
    r"components\Coupons.jsx",
    r"components\Download\AbsentList.jsx",
    r"components\Download\CustomList.jsx",
    r"components\Download\ExamReport.jsx",
    r"components\Download\MonthlyReport.jsx",
    r"components\Download\NumberSheet.jsx",
    r"components\Download\PaidList.jsx",
    r"components\Download\PaymentReport.jsx",
    r"components\Download\PresentList.jsx",
    r"components\Download\UnpaidList.jsx",
    r"components\EditNote.jsx",
    r"components\Exam.jsx",
    r"components\Exams.jsx",
    r"components\ExamsList.jsx",
    r"components\Message.jsx",
    r"components\Note.jsx",
    r"components\OmrHub.jsx",
    r"components\Payment.jsx",
    r"components\PaymentOverview.jsx",
    r"components\ProgramEntry.jsx",
    r"components\ProgramList.jsx",
    r"components\Programs.jsx",
    r"components\StudentDetails.jsx",
    r"components\StudentOverview.jsx",
    r"components\Students.jsx",
    r"components\StudentSite\BannerPics.jsx",
    r"components\StudentSite\BatchTime.jsx",
    r"components\StudentSite\NoticeBoard.jsx",
    r"components\StudentSite\PdfManager.jsx",
    r"components\StudentSite\PromoVideo.jsx",
    r"components\StudentSite\VideoLibrary.jsx",
    r"components\StudentSite\VideoManager.jsx",
    r"components\StuffPart\MyEntry.jsx",
    r"components\StuffPart\StaffDetails.jsx",
    r"components\TakeAttendance.jsx",
    r"components\TakeAttendanceId.jsx",
    r"components\temp.jsx",
    r"components\UpdateStudent.jsx",
    r"components\UserManagement\Modals\PasswordChangeModal.jsx",
    r"components\UserManagement\OfficialsManagement.jsx",
    r"main.jsx",
    r"Provider.jsx"
]

for rel_path in files_to_fix:
    full_path = os.path.join(src_dir, rel_path)
    if not os.path.exists(full_path):
        print(f"Skipping {full_path} - not found")
        continue
    
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if import is already there
    has_import = "import API_URL from " in content or "import { API_URL } from " in content
    
    # Identify depth for import path
    depth = rel_path.count(os.sep)
    import_path = ("../" * depth) + "apiConfig" if depth > 0 else "./apiConfig"
    
    new_content = content.replace("import.meta.env.VITE_API_URL", "API_URL")
    
    if "API_URL" in new_content and not has_import:
        # Add import at the top
        import_line = f"import API_URL from '{import_path}';\n"
        new_content = import_line + new_content
        
    if content != new_content:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {rel_path}")
    else:
        print(f"No changes for {rel_path}")
